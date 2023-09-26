import torch
import torch.nn as nn


class FeedForwardBlock(nn.Module):
    """
    FeedForwardBlock Module:
    
    This class represents the feed-forward network block used within the Transformer's encoder and decoder layers. 
    It consists of two linear transformations with a ReLU activation in between.
    
    The equation implemented is:
    FFN(x) = max(0, xW1 + b1)W2 + b2
    
    Parameters:
    - d_model (int): The dimensionality of input and output. Represents depth of the input feature set.
    - d_ff (int): Dimensionality of the inner-layer. Represents depth of the hidden layer in the feed-forward network.
    - dropout (float): Dropout rate for regularization.
    
    Attributes:
    - linear1 (torch.nn.Linear): First linear transformation layer.
    - dropout (torch.nn.Dropout): Dropout layer for regularization.
    - linear2 (torch.nn.Linear): Second linear transformation layer.
    """

    def __init__(self,d_model:int, d_ff:int, dropout:float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model) # w2 and b2


    def forward(self, x):
        """
        Forward pass for the FeedForwardBlock module.
        
        Parameters:
        - x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
        - torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model)
        """

        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff)
        x = self.linear1(x)
        x = torch.relu(x)

        # Apply dropout for regularization
        x = self.dropout(x)

        # (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        x = self.linear2(x)
        return x
