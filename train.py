import torch
from torch import nn
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from dataset import BilingualDataset,causal_mask
from transformer import build_transformer
from config import get_weights_path, get_config
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter


def get_all_sentences(ds,lang):
    for item in ds:
        yield item['translation'][lang]


def get_or_build_tokenizer(config,ds,lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
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

    # load the dataset
    ds_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')

    # build the tokenizers
    tokenizer_src = get_or_build_tokenizer(config,ds_raw,config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config,ds_raw,config['lang_tgt'])

    # keep 90% of the data for training and 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    # create the training and validation datasets
    ds_train_row, ds_val_row = random_split(ds_raw, [train_ds_size, val_ds_size])
    
    # create the datasets
    train_ds = BilingualDataset(ds_train_row,tokenizer_src,tokenizer_tgt,config['lang_src'],config['lang_tgt'],config['seq_len'])
    val_ds = BilingualDataset(ds_val_row,tokenizer_src,tokenizer_tgt,config['lang_src'],config['lang_tgt'],config['seq_len'])


    max_src_len = 0
    max_tgt_len = 0
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_src_len = max(max_src_len,len(src_ids))
        max_tgt_len = max(max_tgt_len,len(tgt_ids))
    print(f"Maximum length of the source sentence is ={max_src_len}")
    print(f"Maximum length of the target sentence is ={max_tgt_len}")

    train_data_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_data_loader = DataLoader(val_ds, batch_size=1, shuffle=True) # batch_size=1 because we want to evaluate one sentence at a time (not a batch of sentences)

    return train_data_loader, val_data_loader, tokenizer_src, tokenizer_tgt


def get_model(config,vocab_src_len,vocab_tgt_len):
    model = build_transformer(vocab_src_len,
                              vocab_tgt_len,
                              config['seq_len'],
                              config['seq_len'],
                              config['d_model'])
    return model



def train_model(config):
    # define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device is {device}")
    
    # make sure that the model folder exists
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    # get the data loaders
    train_data_loader, val_data_loader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config,tokenizer_src.get_vocab_size(),tokenizer_tgt.get_vocab_size()).to(device)

    # we start tensorboard which allows to visualize the training process
    writer = SummaryWriter(config['exp_name'])


    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])    
    initial_epoch = 0
    global_step = 0

    if config["preload"] is not None:
        model_file = get_weights_path(config, config["preload"])
        print(f"preloading model {model_file}")
        state = torch.load(model_file)
        initial_epoch = state["epoch"]
        global_step = state["global_step"]
        optimizer.load_state_dict(state["optimizer"])
        model.load_state_dict(state["model"])
        global_step = state["global_step"]

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id("[PAD]"),label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch,config['num_epochs']):
        batch_iterator = tqdm(train_data_loader, desc=f"Epoch {epoch}")
        for batch in batch_iterator:
            model.train() # after each time we run the validation, we need to set the model back to train mode
            encoder_input = batch['encoder_input'].to(device) # (batch_size, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (batch_size, seq_len)
            encoder_attention_mask = batch['encoder_attention_mask'].to(device) # (batch_size, 1, 1, seq_len)
            decoder_attention_mask = batch['decoder_attention_mask'].to(device) # (batch_size, 1, seq_len, seq_len)
            label = batch['label'].to(device) # (batch_size, seq_len)
            
            # forward pass
            encoder_output = model.encode(encoder_input, encoder_attention_mask) # (batch_size, seq_len, d_model)
            print("passed encoder")
            decoder_output = model.decode(encoder_output,encoder_attention_mask,decoder_input,decoder_attention_mask) # (batch_size, seq_len, d_model)
            proj_output = model.project(decoder_output) # (batch_size, seq_len, target_vocab_size)

            label = batch['label'].to(device) # (B, seq_len)

            # Compute the loss using a simple cross entropy
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # we run the validation
            run_validation(model,val_data_loader,tokenizer_src,tokenizer_tgt,device,config['seq_len'],lambda msg: batch_iterator.write(msg) ,global_step,writer)


            global_step += 1
        # we save the model after each epoch
        model_file = get_weights_path(config, f'{epoch:02d}')
        print(f"saving model {model_file}")
        torch.save({
            "epoch": epoch,
            "global_step": global_step,
            "optimizer_state_dict": optimizer.state_dict(),
            "model_state_dict": model.state_dict(),
        }, model_file)

def greedy_decode(model,encoder_input,encoder_mask,tokenizer_src,tokenizer_tgt,max_len,device):
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    #precompute the encoder output and reuse it for every token we get from the decoder
    encoder_output = model.encode(encoder_input,encoder_mask) # (batch_size, seq_len, d_model)
    # we start with the sos token
    decoder_input = torch.empty(1,1).fill_(sos_idx).to(device) # (batch_size, token for the decoder input)
    
    while True:
        # the decoder ouput becomes larger than the max_len, we stop
        if decoder_input.shape[1] == max_len:
            break
        # build the decoder attention mask
        decoder_attention_mask = causal_mask(decoder_input.shape[1]).type_as(encoder_input).to(device) # (batch_size, 1, seq_len, seq_len)
        # calculate the decoder output
        decoder_output = model.decode(encoder_output,encoder_mask,decoder_input,decoder_attention_mask)

        # we get the probabilities for the last token (the next token to predict)
        prob = model.project(decoder_output)[:,-1]
        # select the token with the highest probability
        _, next_token = torch.max(prob, dim=1)
        # add the token to the decoder input because it will be used for the next iteration
        decoder_input = torch.cat([decoder_input, torch.empty(1,1).type_as(encoder_input)
                                   .fill_(next_token.item()).to(device)], dim=1)
        # if the token is the eos token, we stop
        if next_token.item() == eos_idx:
            break

    # we return the decoded sentence
    return decoder_input.squeeze(0)


def run_validation(model,
                   validation_ds,
                   tokenizer_src,
                   tokenizer_tgt,
                   device,
                   max_len,
                   print_msg,
                   global_state,
                   writer,
                   num_examples=2):
    model.eval()
    count = 0
    source_sentences = []
    expected_sentences = []
    predicted_sentences = []
    #size of the control window
    console_width = 80
    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_attention_mask = batch['encoder_attention_mask'].to(device)
            assert encoder_input.shape[0] == 1, "only one sentence at a time - Barch size must be 1 for validation" 
            model_output = greedy_decode(model,encoder_input,encoder_attention_mask,tokenizer_src,tokenizer_tgt,max_len,device)
            source_sentence = batch['src_text'][0]
            expected_sentence = batch['tgt_text'][0]
            model_output_sentence = tokenizer_tgt.decode(model_output.detach().cpu().numpy())
            source_sentences.append(source_sentence)
            expected_sentences.append(expected_sentence)
            predicted_sentences.append(model_output_sentence)

            # print to the console
            print_msg("-"*console_width)
            print_msg(f"Source: {source_sentence}") #tqdm writes to the console, so we use print_msg instead of print
            print_msg(f"Expected: {expected_sentence}")
            print_msg(f"Predicted: {model_output_sentence}")
            print_msg("-"*console_width)

            if count == num_examples:
                break
    # log to tensorboard (to be implemented): torchmetrics : character error rate, word error rate, BLEU score


if __name__ == "__main__":
    config = get_config()
    train_model(config)

            


