import torch
import torch.nn as nn
from multiHeadAttentionBlock import MultiHeadAttentionBlock
from feedForwardBlock import FeedForwardBlock
from residualConnection import ResidualConnection

class EncoderBlock(nn.Module):
    """
    EncoderBlock Module:
    
    This class represents a block within the encoder portion of a Transformer model. The EncoderBlock consists of 
    a multi-head self-attention mechanism followed by position-wise fully connected feed-forward network, with 
    residual connections around each of the two sub-layers. 

    Parameters:
    - self_attention_block (MultiHeadAttentionBlock): Instance of MultiHeadAttentionBlock to implement self-attention mechanism.
    - feed_forward_block (FeedForwardBlock): Instance of FeedForwardBlock to implement position-wise feed-forward network.
    - dropout (float): Dropout rate for regularization.

    Attributes:
    - self_attention_block (MultiHeadAttentionBlock): Multi-head self-attention sub-layer.
    - feed_forward_block (FeedForwardBlock): Position-wise feed-forward network sub-layer.
    - residual_connections (nn.ModuleList): List of residual connections. Each residual connection wraps around each sub-layer.
    """

    def __init__(self,
                 self_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock,
                 dropout: float):
        super().__init__()
        
        # Multi-head self-attention sub-layer
        self.self_attention_block = self_attention_block
        
        # Position-wise feed-forward network sub-layer
        self.feed_forward_block = feed_forward_block
        
        # Define two residual connections: one for each sub-layer
        # Using ModuleList to register each residual connection as a submodule of EncoderBlock
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        """
        Forward pass for the EncoderBlock module.
        
        Parameters:
        - x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
        - src_mask (torch.Tensor): Source mask tensor to avoid attending to certain positions.

        Returns:
        - torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        
        # First sub-layer: multi-head self-attention mechanism
        # The input tensor 'x' is used as queries, keys, and values
        # The lambda function is used to match the signature expected by the residual connection
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        
        # Second sub-layer: position-wise feed-forward network
        # The output of the first sub-layer is used as input to the second sub-layer
        x = self.residual_connections[1](x, self.feed_forward_block)
        
        return x
