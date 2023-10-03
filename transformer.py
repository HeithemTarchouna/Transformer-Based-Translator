import torch 
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder
from projectionLayer import ProjectionLayer
from positionalEncoding import PositionalEncoding
from encoderBlock import EncoderBlock
from multiHeadAttentionBlock import MultiHeadAttentionBlock
from feedForwardBlock import FeedForwardBlock
from inputEmbeddings import InputEmbeddings

class Transformer(nn.Module):
    """
    Transformer Module:
    ...
    Methods:
    - encode(src: Tensor, src_mask: Tensor) -> Tensor:
        src: Tensor of shape (batch_size, src_seq_len)
        src_mask: Tensor of shape (batch_size, 1, 1, src_seq_len)
        Returns: Tensor of shape (batch_size, src_seq_len, d_model)
        
    - decode(tgt: Tensor, encoder_output: Tensor, src_mask: Tensor, tgt_mask: Tensor) -> Tensor:
        tgt: Tensor of shape (batch_size, tgt_seq_len)
        encoder_output: Tensor of shape (batch_size, src_seq_len, d_model)
        src_mask: Tensor of shape (batch_size, 1, 1, src_seq_len)
        tgt_mask: Tensor of shape (batch_size, 1, tgt_seq_len, tgt_seq_len)
        Returns: Tensor of shape (batch_size, tgt_seq_len, d_model)
        
    - project(tgt: Tensor) -> Tensor:
        tgt: Tensor of shape (batch_size, tgt_seq_len, d_model)
        Returns: Tensor of shape (batch_size, tgt_seq_len, vocab_size)
    ...
    """
    def encode(self, src, src_mask):
        """
        Perform encoding on source sequence.
        src: Tensor of shape (batch_size, src_seq_len)
        src_mask: Tensor of shape (batch_size, 1, 1, src_seq_len)
        Returns: Tensor of shape (batch_size, src_seq_len, d_model)
        """
        src = self.src_embedding(src)
        src = self.src_positional_encoding(src)
        return self.encoder(src, src_mask)

    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        """
        Perform decoding using encoder outputs and target sequence.
        tgt: Tensor of shape (batch_size, tgt_seq_len)
        encoder_output: Tensor of shape (batch_size, src_seq_len, d_model)
        src_mask: Tensor of shape (batch_size, 1, 1, src_seq_len)
        tgt_mask: Tensor of shape (batch_size, 1, tgt_seq_len, tgt_seq_len)
        Returns: Tensor of shape (batch_size, tgt_seq_len, d_model)
        """
        tgt = self.tgt_embedding(tgt)
        tgt = self.tgt_positional_encoding(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, tgt):
        """
        Project the decoder output to the size of the target vocabulary.
        tgt: Tensor of shape (batch_size, tgt_seq_len, d_model)
        Returns: Tensor of shape (batch_size, tgt_seq_len, vocab_size)
        """
        return self.projection_layer(tgt)
def build_transformer(src_vocab_size:int,
                      tgt_vocab_size:int,
                      src_seq_len:int,
                      tgt_seq_len:int,
                      d_model:int=512,
                      nhead:int=8,
                      num_encoder_layers:int=6,
                      num_decoder_layers:int=6,
                      dropout:float=0.1,
                      d_ff:int=2048):
    """
    Function to build the Transformer model.
    
    Given the hyperparameters, this function initializes all necessary components and returns a Transformer model.
    Additionally, it initializes the weights of the model with Xavier Uniform Initialization for better convergence.
    """
    
    # Encoder
    #---------------------------------------------------------------------------------------
    src_embedding = InputEmbeddings(d_model,src_vocab_size)
    src_positional_encoding = PositionalEncoding(d_model,src_seq_len,dropout)

    encoderBlock = EncoderBlock(MultiHeadAttentionBlock(d_model,nhead,dropout),
                                FeedForwardBlock(d_model,d_ff,dropout),
                                dropout)
    
    encoder_layers = nn.ModuleList([encoderBlock for _ in range(num_encoder_layers)])
    encoder = Encoder(encoder_layers)
    #---------------------------------------------------------------------------------------

    # Decoder
    #---------------------------------------------------------------------------------------
    tgt_embedding = InputEmbeddings(d_model,tgt_vocab_size)
    tgt_positional_encoding = PositionalEncoding(d_model,tgt_seq_len,dropout)
    decoderBlock = EncoderBlock(MultiHeadAttentionBlock(d_model,nhead,dropout),
                                FeedForwardBlock(d_model,d_ff,dropout),
                                dropout)
    
    decoder_layers = nn.ModuleList([decoderBlock for _ in range(num_decoder_layers)])
    decoder = Decoder(decoder_layers)
    #---------------------------------------------------------------------------------------

    # Projection Layer
    #---------------------------------------------------------------------------------------
    projection_layer = ProjectionLayer(d_model,tgt_vocab_size)
    #---------------------------------------------------------------------------------------


    # Transformer
    #---------------------------------------------------------------------------------------
    transformer = Transformer(encoder,decoder,src_embedding,tgt_embedding,src_positional_encoding,tgt_positional_encoding,projection_layer)
    
    #initialize weights
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    
    return transformer

    
