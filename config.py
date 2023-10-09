import os

def get_config():
    return {
        "batch_size": 8,
        'num_epochs': 20,
        'lr': 0.0001,
        'seq_len': 500,
        'd_model': 512,
        'lang_src': 'en',
        'lang_tgt': 'fr',
        'model_folder': 'weights',
        'model_basename': 'transformer_',
        'preload': None,
        'tokenizer_file': 'tokenizer_{0}.json',
        "exp_name": "runs/transformer"
    }

def get_weights_path(config, epoch:str):
    return os.path.join(config['model_folder'], config['model_basename'] + str(epoch) + '.pt')