import torch
import torch.nn as nn
from residualConnection import ResidualConnection
from multiHeadAttentionBlock import MultiHeadAttentionBlock
from feedForwardBlock import FeedForwardBlock

class DecoderBlock(nn.Module):
    """
    DecoderBlock Module:

    This class represents a block within the decoder portion of a Transformer model. Each DecoderBlock
    consists of three sub-layers: a multi-head self-attention mechanism, a multi-head cross-attention 
    mechanism that attends to the encoder’s output, and a position-wise fully connected feed-forward 
    network. Residual connections are employed around each of the sub-layers, followed by layer normalization.

    Parameters:
    - self_attention_block (MultiHeadAttentionBlock): Instance of MultiHeadAttentionBlock for self-attention mechanism.
    - cross_attention_block (MultiHeadAttentionBlock): Instance of MultiHeadAttentionBlock for cross-attention mechanism.
    - feed_forward_block (FeedForwardBlock): Instance of FeedForwardBlock for the position-wise feed-forward network.
    - dropout (float): Dropout rate for regularization.

    Attributes:
    - self_attention_block (MultiHeadAttentionBlock): Multi-head self-attention sub-layer.
    - cross_attention_block (MultiHeadAttentionBlock): Multi-head cross-attention sub-layer.
    - feed_forward_block (FeedForwardBlock): Position-wise feed-forward network sub-layer.
    - residual_connections (nn.ModuleList): List of residual connections, one for each sub-layer.

    Usage:
    The DecoderBlock is used as a building block for constructing the Decoder of the Transformer model.
    """

    def __init__(self,features:int,
                 self_attention_block: MultiHeadAttentionBlock,
                 cross_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock,
                 dropout: float):
        super().__init__()

        # Multi-head self-attention sub-layer
        self.self_attention_block = self_attention_block
        
        # Multi-head cross-attention sub-layer
        self.cross_attention_block = cross_attention_block
        
        # Position-wise feed-forward network sub-layer
        self.feed_forward_block = feed_forward_block
        
        # Define three residual connections: one for each sub-layer
        self.residual_connections = nn.ModuleList([ResidualConnection(features,dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Forward pass for the DecoderBlock module.

        Parameters:
        - x (torch.Tensor): Input tensor of shape (batch_size, target_seq_len, d_model).
        - encoder_output (torch.Tensor): Output tensor from the Encoder of shape (batch_size, source_seq_len, d_model).
        - src_mask (torch.Tensor): Source mask tensor to avoid attending to certain positions in the encoder output.
        - tgt_mask (torch.Tensor): Target mask tensor to avoid attending to future tokens in the target sequence.

        Returns:
        - torch.Tensor: Output tensor of shape (batch_size, target_seq_len, d_model).

        Workflow:
        - The input tensor 'x' is first passed through a self-attention sub-layer, with 'tgt_mask' applied to prevent attending to future tokens.
        - Then, it's passed through a cross-attention sub-layer that attends to the 'encoder_output', with 'src_mask' applied to prevent attending to padded tokens.
        - Finally, the tensor is passed through a position-wise feed-forward network.
        - Each sub-layer is wrapped with a residual connection followed by layer normalization.
        """

        # Self-attention sub-layer with target mask applied
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        
        # Cross-attention sub-layer with source mask applied
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        
        # Position-wise feed-forward network sub-layer
        x = self.residual_connections[2](x, self.feed_forward_block)

        return x
