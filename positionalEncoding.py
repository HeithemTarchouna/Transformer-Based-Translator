import torch 
import torch.nn as nn
import math
class PositionalEncoding(nn.Module):
    
    """
    PositionalEncoding Module:
    
    This class provides positional encodings to input embeddings, enabling the Transformer model to account for
    the order of tokens in a sequence. Positional encodings are designed such that they can be summed with 
    token embeddings. The positional encodings are deterministic and are not updated during training.
    
    Parameters:
    - d_model (int): The dimensionality of the embeddings and positional encodings.
    - seq_len (int): Maximum length of sequences that this model will handle.
    - dropout (float): Dropout rate applied to the sum of embeddings and positional encodings.

    Attributes:
    - d_model (int): The dimensionality of the embeddings and positional encodings.
    - seq_len (int): Maximum length of sequences.
    - dropout (nn.Dropout): Dropout layer applied after adding positional encoding.
    - pe (torch.Tensor): The tensor containing the precomputed positional encodings.
    """
    def __init__(self, d_model:int, seq_len:int, dropout:float ):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(p=dropout)

        # Create constant 'pe' matrix with shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        # create a vector of shape (seq_len, 1) containing values from 0 to seq_len-1
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)

        # PE(pos, 2i) = sin(pos/10000^(2i/d_model))
        # PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))

        # create a vector that contains the values 10000^(2i/d_model) for i = 0, 1, 2, ..., d_model/2
        div_term = torch.exp(torch.arange(0, d_model, 2.0).float() * (-math.log(10000.0) / d_model)) 
        # apply sin to even indices in the array; 2i
        pe[:, 0::2] = torch.sin(position * div_term)
        # apply cos to odd indices in the array; 2i+1
        pe[:, 1::2] = torch.cos(position * div_term)

        # add a batch dimension to the positional encoding (for sentences)
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Forward pass for the PositionalEncoding module.
        
        Parameters:
        - x (torch.Tensor): The input data tensor. Shape (batch_size, seq_len, d_model)

        Returns:
        - torch.Tensor: The input data with positional encodings added, followed by dropout. 
                        Shape remains (batch_size, seq_len, d_model).
        """
        x = x + self.pe[:, :x.size(1), :].requires_grad_(False)
        return self.dropout(x)
