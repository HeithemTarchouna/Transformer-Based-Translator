import torch
import torch.nn as nn
from layerNormalization import LayerNormalization

class ResidualConnection(nn.Module):
    """
    ResidualConnection Module:
    
    This class represents a residual connection followed by a layer normalization, typically used 
    in Transformer models. It allows for the training of deep networks by mitigating the vanishing 
    gradient problem. The input 'x' is first normalized using Layer Normalization, then passed 
    through a sublayer (like a feed-forward network or an attention mechanism). The output of 
    the sublayer is then passed through a dropout for regularization, and finally added to the 
    original input 'x', creating a residual connection.

    Parameters:
    - dropout (float): Dropout rate for regularization.

    Attributes:
    - dropout (nn.Dropout): Dropout layer for regularization.
    - norm (LayerNormalization): Layer normalization.
    """
    def __init__(self,features:int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x))) 