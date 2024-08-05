# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-shakespeare-char'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'shakespeare-char'
wandb_run_name = 'mini-llama'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
max_seq_len = 256 # context of up to 256 previous characters

model_type = "llama"

# baby Llama model :)
n_layers = 6
n_heads = 4
n_kv_heads = n_heads
dim = 128
hidden_dim = 256

# baby gpt model :)
block_size = 256 # context of up to 256 previous characters
dropout = 0.2

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# on macbook and low-end PC also add
device = 'cpu'  # run on cpu only
compile = False # do not torch compile the model
