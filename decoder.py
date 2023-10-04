import torch
import torch.nn as nn
from layerNormalization import LayerNormalization

class Decoder(nn.Module):
    """
    Decoder Module:

    The Decoder class represents the decoder portion of a Transformer model. It is composed of a sequence of 
    DecoderBlocks (specified by 'layers'), each consisting of a self-attention sub-layer, a cross-attention 
    sub-layer, and a feed-forward sub-layer. After the sequence of DecoderBlocks, the output is normalized 
    using Layer Normalization.

    Parameters:
    - layers (nn.ModuleList): A ModuleList containing instances of DecoderBlock that constitute the Decoder.

    Attributes:
    - layers (nn.ModuleList): A ModuleList containing DecoderBlock layers.
    - norm (LayerNormalization): Layer normalization to be applied to the output of the final DecoderBlock.

    Usage:
    This module is used as the decoder in a Transformer model, taking the encoder output and producing the final 
    output sequence.
    """

    def __init__(self,features, layers: nn.ModuleList):
        super().__init__()

        # Sequence of DecoderBlock layers
        self.layers = layers

        # Layer Normalization for the output of the final DecoderBlock
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_masks, tgt_masks):
        """
        Forward pass for the Decoder module.

        Parameters:
        - x (torch.Tensor): Input tensor of shape (batch_size, target_seq_len, d_model).
        - encoder_output (torch.Tensor): Output tensor from the Encoder of shape (batch_size, source_seq_len, d_model).
        - src_masks (torch.Tensor): Source masks tensor to avoid attending to certain positions in the encoder output.
        - tgt_masks (torch.Tensor): Target masks tensor to avoid attending to future tokens in the target sequence.

        Returns:
        - torch.Tensor: Normalized output tensor of shape (batch_size, target_seq_len, d_model).

        Workflow:
        - The input tensor 'x' is passed sequentially through each DecoderBlock in 'layers'.
        - Each DecoderBlock computes self-attention, cross-attention with encoder_output, and applies feed-forward operations, modifying 'x' at each step.
        - The output tensor from the final DecoderBlock is then normalized using 'norm' before being returned.
        """

        # Sequentially pass the input through each DecoderBlock layer with corresponding masks
        for layer in self.layers:
            x = layer(x, encoder_output, src_masks, tgt_masks)

        # Normalize the output using Layer Normalization
        return self.norm(x)
