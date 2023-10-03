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
    
    def __init__(self,
                encoder:Encoder,
                decoder:Decoder,
                src_embedding:InputEmbeddings,
                tgt_embedding:InputEmbeddings,
                src_positional_encoding:PositionalEncoding,
                tgt_positional_encoding:PositionalEncoding,
                projection_layer:ProjectionLayer
                ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        self.src_positional_encoding = src_positional_encoding
        self.tgt_positional_encoding = tgt_positional_encoding
        self.projection_layer = projection_layer



    def encode(self,src,src_mask):
        src = self.src_embedding(src)
        src = self.src_positional_encoding(src)
        src = self.encoder(src,src_mask)
        return src

        

    def decode(self,tgt,encoder_output,src_mask,tgt_mask):
        tgt = self.tgt_embedding(tgt)
        tgt = self.tgt_positional_encoding(tgt)
        tgt = self.decoder(tgt,encoder_output,src_mask,tgt_mask)
        return tgt

    def project(self,tgt):
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

    
