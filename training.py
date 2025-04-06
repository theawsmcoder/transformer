import torch
import torch.nn as nn
from torch.utils.data import dataloader, Dataset, random_split

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace



from pathlib import Path

TOKENIZER_PATH = './saved_data/tokenizer_{0}.json'
SRC_LANG = 'en'
TGT_LANG = 'fr'
TRAIN_DS_SPLIT = 0.9

def get_dataset():
    dataset = load_dataset('opus_books', f'{SRC_LANG}-{TGT_LANG}', split='train')

    # build tokenizer for src and tgt lang
    tokenizer_src = build_tokenizer(dataset, SRC_LANG)
    tokenizer_tgt = build_tokenizer(dataset, TGT_LANG)

    # 90:10 training and validation split
    train_ds_size = int(TRAIN_DS_SPLIT * len(dataset))
    val_ds_size = len(dataset) - train_ds_size

    train_dataset, val_dataset = random_split(dataset, [train_ds_size, val_dataset])

    


def dataset_iterator(ds, lang):
    for item in ds:
        yield item['translation'][lang]


# build tokenizer to convert the words into tokens. a tokenizer for each language
def build_tokenizer(ds, lang):
    tokenizer_path = Path(TOKENIZER_PATH.format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=['[UNK]', '[PAD]', '[SOS]', '[EOS]'], min_frequency=2)
        tokenizer.train_from_iterator(dataset_iterator(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
        return tokenizer
    return Tokenizer.from_file(str(tokenizer_path))
