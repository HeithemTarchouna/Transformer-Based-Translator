import torch
from torch import nn
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split



def get_all_sentences(ds,lang):
    for item in ds:
        yield item['translation'][lang]


def get_or_build_tokenizer(config,ds,lang):
    tokenizer_path = Path(config['tokenizer_path']).format(lang)
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]")) # if word not in vocab, return [UNK]
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds,lang),trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer




def get_ds(config):
    df_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')

    # build the tokenizers
    tokenizer_src = get_or_build_tokenizer(config,df_raw,config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config,df_raw,config['lang_tgt'])

    # keep 90% of the data for training and 10% for validation
    train_ds_size = int(0.9 * len(df_raw))
    val_ds_size = len(df_raw) - train_ds_size
    # create the training and validation datasets
    df_train, df_val = random_split(df_raw, [train_ds_size, val_ds_size])
    
    # create the datasets
     