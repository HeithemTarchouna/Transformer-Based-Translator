import torch
import torch.nn as nn

class LayerNormalization(nn.Module):
    """
    LayerNormalization Module:
    
    This class implements layer normalization, a type of normalization technique which normalizes the outputs 
    of a layer for each given example in a batch independently. Layer normalization can stabilize the training 
    of deep neural networks by reducing internal covariate shift.
    
    Parameters:
    - eps (float): A small value added to the denominator to improve numerical stability. Default is 1e-6.

    Attributes:
    - eps (float): Small constant to prevent division by zero.
    - alpha (torch.nn.Parameter): Scale factor. This is a learnable parameter that scales the normalized output.
    - bias (torch.nn.Parameter): Offset factor. This is a learnable parameter that shifts the normalized output.
    """
    def __init__(self,eps:float = 1e-6,):
        super().__init__()
        self.eps = eps
        # Scale factor (learnable parameter) that will be multiplied to the normalized value
        self.alpha = nn.Parameter(torch.ones(1))         
        # Offset (learnable parameter) that will be added to the normalized value
        self.bias = nn.Parameter(torch.zeros(1)) # nn.paramater makes it a learnable parameter || added

    def forward(self, x):
        """
        Forward pass for the LayerNormalization module.
        
        Parameters:
        - x (torch.Tensor): Input tensor to be normalized.

        Returns:
        - torch.Tensor: Normalized tensor.
        """

        # Calculate the mean and standard deviation along the last dimension of the input tensor
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # Normalize the input tensor and then scale (with alpha) and shift (with bias)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias