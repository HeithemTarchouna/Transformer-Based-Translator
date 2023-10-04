import torch
from torch import nn
from torch.utils.data import Dataset



class BilingualDataset(Dataset):
    def __init__(self,ds,tokenizer_src,tokenizer_tgt,lang_src,lang_tgt,seq_len=512):
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.lang_src = lang_src
        self.lang_tgt = lang_tgt
        self.seq_len = seq_len
        self.sos_token = torch.tensor([self.tokenizer_tgt.token_to_id("[SOS]")],dtype=torch.int)
        self.eos_token = torch.tensor([self.tokenizer_tgt.token_to_id("[EOS]")],dtype=torch.int)
        self.pad_token = torch.tensor([self.tokenizer_tgt.token_to_id("[PAD]")],dtype=torch.int)
        self.unk_token = torch.tensor([self.tokenizer_tgt.token_to_id("[UNK]")],dtype=torch.int)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self,idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.lang_src]
        tgt_text = src_target_pair['translation'][self.lang_tgt]


        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        src_num_pad_tokens = self.seq_len - len(enc_input_tokens)-2 # -2 for sos and eos tokens
        tgt_num_pad_tokens = self.seq_len - len(dec_input_tokens)-1 # -1 for eos token

        if src_num_pad_tokens < 0 or tgt_num_pad_tokens < 0:
            raise Exception("seq_len is too small")
        
        # add sos and eos tokens to encoder
        encoder_input = torch.cat(
            [
            self.sos_token,
            torch.tensor(enc_input_tokens,dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token]*src_num_pad_tokens , dtype=torch.int64)
            ],
            dim=0
        )

        # add eos token to decoder
        decoder_input = torch.cat(
            [
            self.sos_token,
            torch.tensor(dec_input_tokens,dtype=torch.int64),
            torch.tensor([self.pad_token]*tgt_num_pad_tokens , dtype=torch.int64)
            ],
            dim=0
        )
        # add eos token to label (what we expect as output from the decoder)
        label = torch.cat([
            torch.tensor(dec_input_tokens,dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token]*tgt_num_pad_tokens , dtype=torch.int64)], dim=0
        )

        assert len(encoder_input) == self.seq_len
        assert len(decoder_input) == self.seq_len
        assert len(label) == self.seq_len

        return {
            "encoder_input":encoder_input, # (seq_len)
            "decoder_input":decoder_input, # (seq_len)
            # (1,1,seq_len) # 1 for tokens we want to attend to, 0 for tokens we want to ignore (padding)
            # we unsqueeze twice to add the batch and sequence length dimensions
            "encoder_attention_mask":(encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), 

            # we need a causal mask for the decoder attention mask 
            "decoder_attention_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1,1,seq_len) & (1,seq_len,seq_len) = (1,seq_len,seq_len)
            "label":label,# (seq_len)
            "src_text":src_text,
            "tgt_text":tgt_text
            } 
    

def causal_mask(seq_len):
    # create a mask that prevents the decoder from attending to tokens that haven't been generated yet
    # (1,seq_len,seq_len)
    mask = torch.triu(torch.ones(1,seq_len,seq_len).int(), diagonal=1)
    # (1,seq_len,seq_len)
    return mask == 0 # 1 for tokens we want to attend to, 0 for tokens we want to ignore (padding)

        

