# pip install pytorch
import torch
import torch.nn as nn
from torch.nn import functional as F

#  Set random seed for reproducibility
torch.manual_seed(1337)

# Define hyperparameters
batch_size = 39  # Changed from 4 to 39 
block_size = 8
max_iters = 5000
eval_iters = 200
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd=32
#--------

# wget https://raw.githubusercontent.com/TRXAlpha/MysteryMind/main/Training%20Data/murder1.txt
with open('murder1.txt', 'r', encoding ='utf-8') as f:
  text=f.read()

#all unique characters from the input data
chars = sorted (list(set(text)))
vocab_size = len(chars)
#mapping the characters to integers
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  #encoder
decode = lambda l: ''.join([itos[i] for i in l]) #decoder

#--------

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    #   'split' instead of 'split_data' to check the argument
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y

# Get a batch of data
xb, yb = get_batch('train')
#print('inputs:')
#print(xb.shape)
#print(xb)
#print('targets:')
#print(yb.shape)
#print(yb)
#print('----')

# Print context and target for each element
for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t + 1]
        target = yb[b, t]
        print(f"when input is {context.tolist()} the target: {target}")

class Head(nn.Module):

    def __init__(self,head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self,x):
        B,T,C = x.shape
        k = self.key(x) # (B,T,C)
        q = self.query(x) # (B,T,C)
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,C) @ (B,C,T) ---> (B,T,T)
        wei = wei.masked_fill(self.trill[:T,:T] ==0, float ('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim=-1) # (B,T,T
        
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B,T,T) @ (B,T,C) ---> (B,T,C)
        return out

#the bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        #each token directly reads the logits for the next token from a lookup table (database)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_head = Head(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both tensor of integers (B,T)
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arrange(T, device = device)) # (B,T)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.sa_head(x) # applying one head of self-attention
        logits = self.lm_head(tok_emb) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

m = BigramLanguageModel(vocab_size) 
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)


# Define the optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

# Training loop
for steps in range(100):
    # sample a batch of data
    xb, yb = get_batch('train')
    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())
