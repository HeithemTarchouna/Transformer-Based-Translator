import torch 
import torch.nn as nn

class InputEmbeddings(nn.Module):
    """
    InputEmbeddings Module:
    
    This class handles the initial embedding of input tokens into a continuous space of dimensionality d_model. 
    It also scales the output by the square root of d_model to make the model's training more stable.

    Parameters:
    - d_model (int): The dimensionality of the output embeddings.
    - vocab_size (int): The number of unique tokens in the vocabulary. This determines the number of rows in the embedding matrix.

    Attributes:
    - d_model (int): The dimensionality of the output embeddings.
    - vocab_size (int): The number of unique tokens in the vocabulary.
    - embedding (nn.Embedding): The embedding layer used to map tokens into a continuous space.
    """

    def __init__(self, d_model:int, vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model)

    def forward(self, x):
        return self.embedding(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
    