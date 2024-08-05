import math
import inspect
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

"""
A shrinked Llama2 model
"""


@dataclass
class LlamaConfig:
    dim: int = 256
    hidden_dim: int = 512
    vocab_size: int = None
    n_layers: int = 6
    n_heads: int = 4
    n_kv_heads: int = 4
    bias : bool = False
    max_seq_len: int = 256
    max_batch_size: int = 64

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim = True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x).type_as(x)
        return output * self.weight

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]) # x: (bsz, seq_len, n_heads, head_dim / 2)
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)] # (1, seq_len, 1, head_dim / 2)
    return freqs_cis.view(*shape)

def rope(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.



    """
    xq_ = torch.view_as_complex(xq.reshape(*xq.shape[:-1], -1, 2)) # (bsz, seq_len, n_heads, head_dim / 2, 2)
    xk_ = torch.view_as_complex(xk.reshape(*xk.shape[:-1], -1, 2)) # (bsz, seq_len, n_heads, head_dim / 2, 2)
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_) # (1, seq_len, 1, head_dim / 2)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3) # ???
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim) # (bsz, seqlen, n_heads, head_dim)
    )

class Attention(nn.Module):
    """Multi-head attention module."""
    def __init__(self, args: LlamaConfig):
        """
        Initialize the Attention module.

        Args:
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_kv_heads (int): Number of key and value heads.
            n_heads (int): Number of local query heads.
            n_rep (int): Number of repetitions for local heads.
            head_dim (int): Dimension size of each attention head.
            wq (ColumnParallelLinear): Linear transformation for queries.
            wk (ColumnParallelLinear): Linear transformation for keys.
            wv (ColumnParallelLinear): Linear transformation for values.
            wo (RowParallelLinear): Linear transformation for output.
            cache_k (torch.Tensor): Cached keys for attention.
            cache_v (torch.Tensor): Cached values for attention.

        """
        super().__init__()

        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads = args.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias = False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias = False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias = False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias = False)

        self.cache_k = torch.zeros(
            (
                args.max_batch_size,  # B x S x T
                args.max_seq_len,
                self.n_kv_heads,
                self.head_dim,
            )
        )

        self.cache_v = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_kv_heads,
                self.head_dim,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for caching.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.
            mask (torch.Tensor, optional): Attention mask tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim) # (batch, seqlen, n_heads, head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim) # (batch, seqlen, n_heads, head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim) # (batch, seqlen, n_heads, head_dim)

        # apply rotatary position relative encoding
        xq, xk = rope(xq, xk, freqs_cis)

        # add k and v to the kvcache
        self.cache_k[:bsz, start_pos:seqlen] = xk
        self.cache_v[:bsz, start_pos:seqlen] = xv

        keys = self.cache_k[:bsz, :seqlen]   # (batch, cache_len + seqlen, n_heads, head_dim)
        values = self.cache_v[:bsz, :seqlen] # (batch, cache_len + seqlen, n_heads, head_dim)

        # compute attenion
        repeat_kv(keys, self.n_rep)
        repeat_kv(values, self.n_rep)

        xq = xq.transpose(1, 2) # (batch, n_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2) # (batch, n_heads, cache_len + seqlen, head_dim)
        values = values.transpose(1, 2) # (batch, n_heads, cache_len + seqlen, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim) # attention (batch, n_heads, seqlen, cache_len + seqlen)
        if mask is not None:
            scores = scores + mask
        scores = torch.softmax(scores, dim = -1).type_as(xq)

        # weighed sum of values
        output = torch.matmul(scores, values) # (batch, n_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # the output of attention layer
        return self.wo(output)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden dimension. Defaults to None.

        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third layer.

        """
        super().__init__()

        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(
            dim, hidden_dim, bias = False,
        )
        self.w2 = nn.Linear(
            hidden_dim, dim, bias=False,
        )
        self.w3 = nn.Linear(
            dim, hidden_dim, bias=False,
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class LlamaLayer(nn.Module):
    def __init__(self, layer_id: int, args: LlamaConfig):
        """
        Initialize a TransformerBlock.

        Args:
            layer_id (int): Identifier for the layer.
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            layer_id (int): Identifier for the layer.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.

        """
        super().__init__()
        self.n_heads   = args.n_heads
        self.dim       = args.dim
        self.head_dim  = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim = args.dim,
            hidden_dim = args.hidden_dim,
            multiple_of = 1,
            ffn_dim_multiplier = None
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim)
        self.ffn_norm = RMSNorm(args.dim)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for attention caching.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
            mask (torch.Tensor, optional): Masking tensor for attention. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        h = x + self.attention(
            self.attention_norm(x), start_pos, freqs_cis, mask
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs)  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

class Llama(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.dim is not None

        self.config = config

        self.transformer = nn.ModuleDict(dict(
            tok_embd = nn.Embedding(self.config.vocab_size, self.config.dim),
            layers   = nn.ModuleList([LlamaLayer(index, config) for index in range(config.n_layers)]),
            norm     = RMSNorm(self.config.dim)
            ))

        self.output  = nn.Linear(self.config.dim, self.config.vocab_size, bias = False),

        # pre-compute rope frequencies
        self.freqs_cis = precompute_freqs_cis(
            # Note that self.params.max_seq_len is multiplied by 2 because the token limit for the Llama 2 generation of models is 4096.
            # Adding this multiplier instead of using 4096 directly allows for dynamism of token lengths while training or fine-tuning.
            self.config.dim // self.config.n_heads, self.config.max_seq_len * 2
        )

        self.apply(self._init_weights)

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor = None, start_pos = 0):
        _, seqlen = tokens.shape # (batch, seqlen)

        assert start_pos >= 0 and start_pos < self.config.max_seq_len
        assert seqlen <= self.config.max_seq_len

        h = self.transformer.tok_embd(tokens) # (batch, seqlen, dim)

        # rope frequencies for this batch
        self.freq_cis = self.freqs_cis.to(h.device) # (seqlen, dim / 2) complex
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        #if seqlen > 1:
        #    mask = torch.full(
        #        (seqlen, seqlen), float("-inf"), device=tokens.device
        #    )

        #    mask = torch.triu(mask, diagonal=1)

        #    # When performing key-value caching, we compute the attention scores
        #    # only for the new sequence. Thus, the matrix of scores is of size
        #    # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
        #    # j > cache_len + i, since row i corresponds to token cache_len + i.
        #    mask = torch.hstack([
        #        torch.zeros((seqlen, start_pos), device=tokens.device),
        #        mask
        #    ]).type_as(h)

        for layer in self.transformer.layers:
            h = layer(h, start_pos, freqs_cis, mask)

        h = self.transformer.norm(h)

        o = self.output(h)

        logits = o

        # Training time
        if targets is not None:
            # logits.view(-1, logits.size(-1)), (batch * seqlen, vocab_size)
            # targets.view(-1), (batch * seqlen, ), torch uses index instead one hot encoding to represent target.
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index = -1)
        else:
            loss = None

        return logits, loss


    def get_num_params(self):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    #def crop_block_size(self, max_seq_len):
    #    # model surgery to decrease the block size if necessary
    #    # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
    #    # but want to use a smaller block size for some smaller, simpler model
    #    assert max_seq_len <= self.config.max_seq_len
    #    self.config.max_seq_len = max_seq_len
    #    self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
    #    for block in self.transformer.h:
    #        if hasattr(block.attn, 'bias'):
    #            block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]
