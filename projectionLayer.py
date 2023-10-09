import torch
import torch.nn as nn

class ProjectionLayer(nn.Module):
    """
    ProjectionLayer Module:

    The ProjectionLayer class is designed to transform the output tensor of the Transformer's decoder to 
    the final output tensor of shape (batch_size, seq_len, vocab_size) using a linear layer. This tensor can 
    be interpreted as unnormalized log probabilities of tokens from the vocabulary, which can be converted to 
    probabilities using softmax.

    Parameters:
    - d_model (int): The dimensionality of the input tensor (typically the dimensionality of the model's embeddings).
    - vocab_size (int): The size of the vocabulary, which determines the output tensor's last dimension.

    Attributes:
    - projection_head (nn.Linear): Linear layer used for projecting the input tensor to the output tensor of specified vocab_size.

    Usage:
    - Used at the final stage of the Transformer's decoder to project its outputs to log probabilities of tokens from the vocabulary.

    Note:
    - Ensure to use a log_softmax function following this layer when calculating the loss during training.
    """

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()

        # Define the linear layer used for the projection
        # The input dimension is d_model, and the output dimension is vocab_size
        self.projection_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """
        Forward pass for the ProjectionLayer module.

        Parameters:
        - x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
        - torch.Tensor: Output tensor of shape (batch_size, seq_len, vocab_size), representing unnormalized log probabilities of tokens.

        Workflow:
        - The input tensor 'x' is passed through the linear 'projection_head' layer, projecting it to 'vocab_size' dimensions.
        - Log softmax is applied to the result across the last dimension to obtain log probabilities.
        """

        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, vocab_size)
        # Note: Log softmax is applied here to obtain log probabilities, ensure to use it accordingly in the loss calculation
        #return torch.nn.functional.log_softmax(self.projection_head(x), dim=-1)
        return self.projection_head(x)