import torch
import torch.nn as nn
import torch.optim.adam
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path
from tqdm import tqdm

from dataset import BilingualDataset, causal_mask
from models import build_transformer

TOKENIZER_PATH = 'tokenizer_{0}.json'
SRC_LANG = 'en'
TGT_LANG = 'fr'
TRAIN_DS_SPLIT = 0.9
SEQ_LEN = 360
BATCH_SIZE = 8
LR = 1e-4
DMODEL = 512
MODEL_FOLDER = 'weights'
MODEL_BASENAME = 'model_{SRC_LANG}_{TGT_LANG}'
PRELOAD = None
EXPERIMENT_NAME = 'runs/model'
EPOCHS = 20


def get_weights_file_path(epoch: str):
    model_filename = MODEL_BASENAME + '_{epoch}.pt'
    return str(Path('.') / MODEL_FOLDER / model_filename)


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


def get_dataset():
    dataset = load_dataset('opus_books', f'{SRC_LANG}-{TGT_LANG}', split='train')

    # build tokenizer for src and tgt lang
    tokenizer_src = build_tokenizer(dataset, SRC_LANG)
    tokenizer_tgt = build_tokenizer(dataset, TGT_LANG)

    # 90:10 training and validation split
    train_ds_size = int(TRAIN_DS_SPLIT * len(dataset))
    val_ds_size = len(dataset) - train_ds_size

    train_dataset_raw, val_dataset_raw = random_split(dataset, [train_ds_size, val_ds_size])

    src_max_len = 0
    tgt_max_len = 0
    for item in dataset:
        src_ids = tokenizer_src.encode(item['translation'][SRC_LANG]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][TGT_LANG]).ids
        src_max_len = max(src_max_len, len(src_ids))
        tgt_max_len = max(tgt_max_len, len(tgt_ids))

    print("Max length for source sentence is: ", src_max_len)
    print("Max length for target sentence is: ", tgt_max_len)

    train_dataset = BilingualDataset(train_dataset_raw, tokenizer_src, tokenizer_tgt, SRC_LANG, TGT_LANG, SEQ_LEN)
    val_dataset = BilingualDataset(val_dataset_raw, tokenizer_src, tokenizer_tgt, SRC_LANG, TGT_LANG, SEQ_LEN)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def build_model(src_vocab_size, tgt_vocab_size):
    return build_transformer(src_vocab_size, tgt_vocab_size, seq_length=SEQ_LEN)


def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('using device:', device)

    Path(MODEL_FOLDER).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_dataset()

    model = build_model(tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    #Tensorboard
    writer = SummaryWriter(EXPERIMENT_NAME)

    optimiser = torch.optim.Adam(model.parameters(), lr=LR, eps=1e-9)

    initial_epoch = 0
    global_step = 0 # used for tensorboard to keep track of global step

    if PRELOAD:
        model_filename = get_weights_file_path(PRELOAD)
        print(f"preloading from file: {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimiser.load_state_dict(state['optimiser_state_dict'])
        global_step = state['global_step']

    # label smoothing makes the model to be less sure of its choices and makes it more accurate by reducing overfitting. here it will take 0.1% value of its highest probable score and distribute it to others
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device) 

    for epoch in range(initial_epoch, EPOCHS):
        model.train()

        batch_iterator = tqdm(train_dataloader, desc=f"processing epoch {epoch:02d}")

        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device) # (N, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (N, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (N, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (N, 1, seq_len, seq_len)

            encoder_output = model.encode(encoder_input, encoder_mask) # (N, seq_len, d_model)
            decoder_output = model.decode(decoder_input, encoder_output, decoder_mask, encoder_mask) # (N, seq_len, d_model)
            projection_output = model.project(decoder_output) # (N, seq_len, vocab_size)

            label = batch['label'].to(device) # (N, seq_len)

            # (N, seq_len, vocab_size) -> # (N * seq_len, vocab_size) for proojection output
            loss = loss_fn(projection_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))

            batch_iterator.set_postfix(f"loss: {loss.item():6.3f}")

            # log the loss
            writer.add_scalar("train_loss", loss.item(), global_step)
            writer.flush()

            # backpropogate
            loss.backward()

            # optimiser step to update the weights
            optimiser.step()
            optimiser.zero_grad()

            global_step += 1

        # save model each epoch
        model_filename = get_weights_file_path(f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimiser_state_dict': optimiser.state_dict(),
            'global_step': global_step
        }, model_filename)





if __name__ == '__main__':
    train_model()