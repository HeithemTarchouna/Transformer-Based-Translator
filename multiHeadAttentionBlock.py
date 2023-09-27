import torch
import torch.nn as nn

class MultiHeadAttentionBlock(nn.Module):
    """
    MultiHeadAttentionBlock Module:

    This class represents the multi-head attention block used within the Transformer's encoder and decoder layers.
    The multi-head mechanism allows the model to focus on different words for a given input word. 

    Parameters:
    - d_model (int): The dimensionality of input and output. Represents depth of the input feature set.
    - h (int): Number of attention heads.
    - dropout (float): Dropout rate for regularization.
    
    Attributes:
    - d_model (int): The dimensionality of input and output.
    - h (int): Number of attention heads.
    - d_k (int): Dimensionality of queries and keys (d_model // h).
    - w_q (nn.Linear): Linear layer for transforming queries.
    - w_k (nn.Linear): Linear layer for transforming keys.
    - w_v (nn.Linear): Linear layer for transforming values.
    - w_o (nn.Linear): Final linear layer for producing the output.
    - dropout (nn.Dropout): Dropout layer for regularization.
    """

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
        """
        Static method for calculating attention weights and scores.
        
        Parameters:
        - query (torch.Tensor): Query tensor of shape (batch_size, h, seq_len, d_k).
        - key (torch.Tensor): Key tensor of shape (batch_size, h, seq_len, d_k).
        - value (torch.Tensor): Value tensor of shape (batch_size, h, seq_len, d_k).
        - mask (torch.Tensor): Mask to avoid attending to certain positions.
        - dropout (nn.Dropout): Dropout layer for regularization.
        
        Returns:
        - Tuple[Torch.Tensor]: Tuple containing output tensor and attention scores.
        """

        d_k = query.shape[-1]

        # Compute attention scores using scaled dot-product attention mechanism
        # (batch_size, h, seq_len, d_k) @ (batch_size, h, d_k, seq_len) -> (batch_size, h, seq_len, seq_len)
        attention_scores = query @ key.transpose(-2,-1) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

        # if we don't want some words to be attended to (to interact), we set their attention scores to -1e9 (representing -infinity)
        # so that when we apply softmax, they will be 0
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask==0, -1e9)

        # (batch_size, h, seq_len, seq_len,seq_len)
        # Apply softmax to obtain attention distribution
        attention_scores = nn.Softmax(dim=-1)(attention_scores)

        # Apply dropout for regularization
        if droput is not None:
            attention_scores = droput(attention_scores)
        
        return (attention_scores @ value), attention_scores




    def forward(self,q,k,v,mask=None):
        """
        Forward pass for the MultiHeadAttentionBlock module.
        
        Parameters:
        - q (torch.Tensor): Query tensor of shape (batch_size, seq_len, d_model).
        - k (torch.Tensor): Key tensor of shape (batch_size, seq_len, d_model).
        - v (torch.Tensor): Value tensor of shape (batch_size, seq_len, d_model).
        - mask (torch.Tensor): Mask to avoid attending to certain positions.
        
        Returns:
        - torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """

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

 