import torch 
import torch.nn as nn
from torch.nn import functional as F


class Block(nn.Module):
    #'Transformer Block'
    def __init__(self,embed_dim,heads,block_size,p):
        super().__init__()
        head_size = embed_dim//heads
        self.sa = MultiHeadedAttention(num_heads = heads,head_size=head_size,p=p,block_size=block_size)
        self.ffwd = FeedForward(embed_dim,p)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
    def forward(self,x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
class Transformer(nn.Module):
  def __init__(self,vocab_size,embed_dim=32,block_size=8,n_layer=6,heads=6,dropout=0.2,device='cpu'):
    super(Transformer,self).__init__()
    # each token directly reads off the logits for the next token from a lookup table 
    self.block_size = block_size
    self.embeddings = nn.Embedding(vocab_size,embed_dim)
    self.positional_embeddings = nn.Embedding(block_size,embed_dim)
    self.blocks = nn.Sequential(
        *[Block(embed_dim=embed_dim,heads=heads,block_size=block_size,p=dropout) for _ in range(n_layer)]
    ) #this is alright, no problem here 
    self.lm_head = nn.Linear(embed_dim,vocab_size)
    # self.mlp = FeedForward(embed_dim)
    self.device = device
  
  def forward(self,idx,targets=None):
    # idx and targets are both (B,T) tensor of integers
    B,T = idx.shape
    tok_emb = self.embeddings(idx) #(B,T,C)
    pos_emb = self.positional_embeddings(torch.arange(T,device=self.device)) #(T,C)
    x = tok_emb+pos_emb # (B,T,C)
    x = self.blocks(x)
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

# class Head(nn.Module):
#   def __init__(self,head_size,p,embed_dim=32,block_size=8):
#     super().__init__()
#     self.key = nn.Linear(embed_dim,head_size,bias=False)
#     self.query = nn.Linear(embed_dim,head_size,bias=False)
#     self.value = nn.Linear(embed_dim,head_size,bias=False)
#     self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))

#     self.dropout = nn.Dropout(p)
#   def forward(self,x):
#     B,T,C = x.shape
#     key = self.key(x)
#     query = self.query(x)
#     wei = query @ key.transpose(-2,-1) * C**-0.5
#     wei =wei.masked_fill(self.tril[:T,:T]==0,float("-inf"))
#     wei = F.softmax(wei,dim=-1)
#     wei = self.dropout(wei)
#     v = self.value(x)
#     out = wei @ v
#     return out 

# class MultiHeadedAttention(nn.Module):
#   def __init__(self,num_heads,head_size,p,block_size=8):
#     super().__init__()
#     self.heads = nn.ModuleList([Head(head_size=head_size,p=p,embed_dim=num_heads*head_size,block_size=block_size) for _ in range(num_heads)])
#     ## parallelize this shit 
#     self.proj = nn.Linear(num_heads*head_size,num_heads*head_size)
#     self.dropout = nn.Dropout(p)
#   def forward(self,x):
#     out = torch.cat([h(x) for h in self.heads],dim=-1)
#     out = self.proj(out)
#     out = self.dropout(out)
#     return out 

class MultiHeadedAttention(nn.Module):
  def __init__(self,num_heads,head_size,p,block_size=8):
    super().__init__()
    self.num_heads = num_heads
    self.head_size = head_size
    self.block_size = block_size
    self.embed_dim = self.num_heads*self.head_size 

    #since self.embed_dim = self.num_heads*self.head_size
    self.key = nn.Linear(self.embed_dim,self.embed_dim,bias=False)
    self.query = nn.Linear(self.embed_dim,self.embed_dim,bias=False)
    self.value = nn.Linear(self.embed_dim,self.embed_dim,bias=False)

    self.register_buffer('tril',torch.tril(torch.ones(self.block_size,self.block_size)))
    self.dropout = nn.Dropout(p)
    self.proj = nn.Linear(self.embed_dim,self.embed_dim)

  def forward(self,x):
    B,T,C = x.shape
    # C is the embedding dimension 
    key = self.key(x).view(B,T,self.num_heads,self.head_size).transpose(1,2)
    query = self.query(x).view(B,T,self.num_heads,self.head_size).transpose(1,2)
    value = self.value(x).view(B,T,self.num_heads,self.head_size).transpose(1,2)
    ## B,num_heads,TimeStep,Embeded Dimension

    wei = (query @ key.transpose(-2,-1)) * (self.head_size ** -0.5)
    wei = wei.masked_fill(self.tril[:T,:T]==0,float("-inf"))
    wei = F.softmax(wei,dim=-1)
    wei = self.dropout(wei)

    out = wei @ value # B,num_heads,T,head_size
    out = out.transpose(1,2).contigous().view(B,T,C) ## B,T,num_heads*head_size
    out = self.proj(out)
    return out 



class FeedForward(nn.Module):
  def __init__(self,embed_dim,p):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(embed_dim,4*embed_dim),
      nn.ReLU(),
      nn.Linear(4*embed_dim,embed_dim),
      nn.Dropout(p)
    )
  def forward(self,x):
    return self.net(x)


