import torch 
import random 

class batched_sampling():
  def __init__(self,train_data,val_data,batch_size=4,block_size=8,seed=1337):
    self.train_data = train_data
    self.val_data = val_data
    self.batch_size = batch_size
    self.block_size  = block_size
    torch.manual_seed(seed)
  
  def get_batch(self,split):
    '''
    returns a batch of batch_size 
    with block_size number of tokens 
    a total of batch_size*block_size number of independent samples 
    '''
    data = self.train_data if split.lower() == 'train' else self.val_data
    ix = torch.randint(len(data)-self.block_size, (self.batch_size,))
    x = torch.stack([data[i:i+self.block_size] for i in ix]) # inputs 
    y = torch.stack([data[i+1:i+self.block_size+1] for i in ix]) # targets (offset by 1 for x)
    return x,y 
