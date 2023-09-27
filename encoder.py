import torch
import torch.nn as nn
from layerNormalization import LayerNormalization

class Encoder(nn.Module):
    """
    Encoder Module:

    This class represents the Encoder part of a Transformer model. The Encoder consists of a sequence of identical
    layers (EncoderBlock), each with a multi-head self-attention mechanism and a position-wise feed-forward network. 
    After passing through the sequence of EncoderBlocks, the output is normalized using Layer Normalization.

    Parameters:
    - layers (nn.ModuleList): A ModuleList containing instances of EncoderBlock that make up the Encoder.

    Attributes:
    - layers (nn.ModuleList): A ModuleList containing the EncoderBlock layers.
    - norm (LayerNormalization): Layer normalization to be applied to the output of the final EncoderBlock.

    Usage:
    This module is used as the encoder in a Transformer model, processing the input sequence before it's fed into the decoder.
    """

    def __init__(self, layers: nn.ModuleList):
        super().__init__()

        # Sequence of EncoderBlock layers
        self.layers = layers

        # Layer Normalization for the output of the final EncoderBlock
        self.norm = LayerNormalization()

    def forward(self, x, src_mask):
        """
        Forward pass for the Encoder module.

        Parameters:
        - x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
        - src_mask (torch.Tensor): Source mask tensor to avoid attending to certain positions.

        Returns:
        - torch.Tensor: Normalized output tensor of shape (batch_size, seq_len, d_model).

        Workflow:
        - The input tensor 'x' is passed sequentially through each EncoderBlock in 'layers'.
        - Each EncoderBlock computes self-attention and applies feed-forward operations, modifying 'x' at each step.
        - The output tensor from the final EncoderBlock is then normalized using 'norm' before being returned.
        """

        # Sequentially pass the input through each EncoderBlock layer
        for layer in self.layers:
            x = layer(x, src_mask)

        # Normalize the output using Layer Normalization
        return self.norm(x)
