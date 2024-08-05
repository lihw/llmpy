import os
import time
import math
import pickle
import json
from contextlib import nullcontext

import numpy as np
import torch

from llama import Llama, LlamaConfig
from gpt2 import GPT, GPTConfig
from model import LLAMA

# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'

# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())

# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 4 # if gradient_accumulation_steps > 1, this is the micro-batch size

# model
model_type = "gpt"

# llama model
n_layers = 12
n_heads = 12
n_kv_heads = 12
max_seq_len = 256
dim = 768
hidden_dim = dim * 2
bias = False # do we use bias inside LayerNorm and Linear layers?

# gpt2 model
block_size = 1024
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+

# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0

# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
compile = True # use PyTorch 2.0 to compile the model to be faster
dtype = 'bfloat16'

# collect all parameters into a dict
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]

# read specific configuration file
exec(open('configurator.py').read()) # overrides from command line or config file

## FIXME: use fp16 for training???
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler

config = {k: globals()[k] for k in config_keys} # will be useful for logging

# print batch size
tokens_per_iter = gradient_accumulation_steps * batch_size * dim
print(f"tokens per iteration will be: {tokens_per_iter:,}")
#  print all configuration
print(json.dumps(config, sort_keys=False, indent=4))

# the global var doesn't sync with config file
device = config["device"]

os.makedirs(out_dir, exist_ok = True)
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
print(f"dtype: {dtype}")

data_dir = os.path.join('data', dataset)

def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - config["max_seq_len"], (batch_size,)) # (batch_size, ) of uint32
    x = torch.stack([torch.from_numpy((data[i : i + config["max_seq_len"]]).astype(np.int64)) for i in ix]) #(batch_size, max_seq_len)
    y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + config["max_seq_len"]]).astype(np.int64)) for i in ix]) #(batch_size, max_seq_len)
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking = True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init vocab
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# init model
if model_type == "llama":
    model_args = dict(n_layers = n_layers,
            dim = dim,
            hidden_dim = hidden_dim,
            vocab_size = None,
            n_heads = n_heads,
            n_kv_heads = n_heads,
            bias = bias,
            max_seq_len = max_seq_len,
            max_batch_size = batch_size,
            ) # start with model_args from command line
else: #gpt2
    model_args = dict(n_layer = n_layers,
            n_head = n_heads,
            n_embd = dim,
            block_size = block_size,
            bias = bias,
            vocab_size = None,
            dropout=dropout) # start with model_args from command line

if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304

    if model_type == "llama":
        llamaconf = LlamaConfig(**model_args)
        model = Llama(llamaconf)
    else: # gpt2
        config1 = {"vocab_size": meta_vocab_size,
              "n_head": n_heads,
              "hidden_size": dim,
              "n_layer": n_layers,
              "n_embd": dim,
              "n_local_heads": n_heads,
              "n_local_kv_heads": n_heads,
              "eps": 1e-6,
              "max_len": max_seq_len,
              "rope_theta": 1.0,
              "num_key_value_heads": n_heads,
              "attention_dropout": 0.25,
              "rms_norm_eps": 1.0,
              "weight_decay": 0.1,
              "block_size": max_seq_len}
        model = LLAMA(config1)
        #gpt2conf = GPTConfig(**model_args)
        #model = GPT(gpt2conf)

#if dim < model.config.dim:
#    model.crop_block_size(dim)
#    model_args['block_size'] = dim # so that the checkpoint will have the right value

model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter = 0
best_val_loss = 1e9

X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter = 0 # number of iterations in the lifetime of this process
running_mfu = -1.0
while True:
    lr = get_lr(iter) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # micro batch (forward + backward) to accumulate gradients
    for _ in range(gradient_accumulation_steps):
        with ctx:
            # forward pass
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps

        # next micro_batch
        X, Y = get_batch('train')

        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()

    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # update model parameters by stepping the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()

    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter % log_interval == 0:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        #if local_iter >= 5: # let the training loop settle a bit
            #mfu = model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            #running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter += 1
    local_iter += 1

    # termination conditions
    if iter > max_iters:
        break



