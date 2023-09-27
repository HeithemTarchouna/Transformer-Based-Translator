import torch
import torch.nn as nn

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model:int, h:int, dropout:float):
        super().__init__()
        self.d_model = d_model
        self.h = h 
        assert d_model % h == 0, "d_model must be divisible by h"
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model) # dv = dk = d_model // h
        self.dropout = nn.Dropout(dropout)

    @staticmethod   
    def attention(query,key,value,mask,droput:nn.Dropout):
        d_k = query.shape[-1]

        # (batch_size, h, seq_len, d_k) @ (batch_size, h, d_k, seq_len) -> (batch_size, h, seq_len, seq_len)
        attention_scores = query @ key.transpose(-2,-1) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

        # if we don't want some words to be attended to (to interact), we set their attention scores to -1e9 (representing -infinity)
        # so that when we apply softmax, they will be 0
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask==0, -1e9)

        # (batch_size, h, seq_len, seq_len,seq_len)
        attention_scores = nn.Softmax(dim=-1)(attention_scores)
        if droput is not None:
            attention_scores = droput(attention_scores)
        
        return (attention_scores @ value), attention_scores




    def forward(self,q,k,v,mask=None):

        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)


        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, h, d_k) -> (batch_size, h, seq_len, d_k)
        # split into h heads along embedding dimension (d_model)
        # we transpose because we each head to see all the words in the sequence (seq_len x d_k) but only through dk dimensions
        # (ie : smaller part of the embedding)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2) 
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2) 
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2) 



        x,self.attention_scores = MultiHeadAttentionBlock.attention(query,key,value,mask,self.dropout)

        # (batch_size, h, seq_len, d_k) -> (batch_size, seq_len, h, d_k) -> (batch_size, seq_len, d_model)
        x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k) # contiguous() is needed because of the view() operation and d_model is self.h * self.d_k

        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model) 
        return self.w_o(x)

 