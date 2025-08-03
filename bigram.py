import torch 
import torch.nn as nn
from torch.nn import functional as F

class BigramModel(nn.Module):
  def __init__(self,vocab_size,embed_dim=32,block_size=8,device='cpu'):
    super(BigramModel,self).__init__()
    # each token directly reads off the logits for the next token from a lookup table 
    self.block_size = block_size
    self.embeddings = nn.Embedding(vocab_size,embed_dim)
    self.positional_embeddings = nn.Embedding(block_size,embed_dim)
    self.sa_head = Head(head_size=embed_dim)
    self.lm_head = nn.Linear(embed_dim,vocab_size)
    self.device = device
  
  def forward(self,idx,targets=None):
    # idx and targets are both (B,T) tensor of integers
    B,T = idx.shape
    tok_emb = self.embeddings(idx) #(B,T,C)
    pos_emb = self.positional_embeddings(torch.arange(T,device=self.device)) #(T,C)
    x = tok_emb+pos_emb # (B,T,C)
    x = self.sa_head(x)
    logits = self.lm_head(x)
    if targets is None:
      loss = None
    else:
      B,T,C = logits.shape
      logits = logits.view(B*T,C)
      targets  = targets.view(B*T)
      loss = F.cross_entropy(logits,targets) # wants (B,C,T)
      # predicting what comes next, based on the last token 
    return logits,loss 

  @torch.no_grad()
  def generate(self,idx,max_len):
    # idx = (B,TimeStep)
    for _ in range(max_len):
      idx_cond = idx[:,-self.block_size:]
      logits,loss = self(idx_cond)
      #get the last time step 
      logits = logits[:, -1, :] #becomes (B,C)
      #apply softmax for probabilities over logits. 
      probs = F.softmax(logits,dim=1)
      #sample from the probabilites
      idx_next = torch.multinomial(probs,num_samples=1)
      ##append the next sample to the output 
      idx = torch.cat((idx,idx_next),dim=1) #(B,T+1)
    return idx

class Head(nn.Module):
  def __init__(self,head_size,embed_dim=32,block_size=8):
    super().__init__()
    self.key = nn.Linear(embed_dim,head_size,bias=False)
    self.query = nn.Linear(embed_dim,head_size,bias=False)
    self.value = nn.Linear(embed_dim,head_size,bias=False)
    self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))

  def forward(self,x):
    B,T,C = x.shape
    key = self.key(x)
    query = self.query(x)

    wei = query @ key.transpose(-2,-1) * C**-0.5
    wei =wei.masked_fill(self.tril[:T,:T]==0,float("-inf"))
    wei = F.softmax(wei,dim=-1)
    v = self.value(x)
    out = wei @ v
    return out 

