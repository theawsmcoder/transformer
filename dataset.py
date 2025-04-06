import torch
from torch import nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self, dataset, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_length):
        super(BilingualDataset).__init__()

        self.dataset = dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_length = seq_length

        self.sos_token = torch.Tensor([tokenizer_src.token_to_id(['[SOS]'])], dtype = torch.int64)
        self.eos_token = torch.Tensor([tokenizer_src.token_to_id(['[EOS]'])], dtype = torch.int64)
        self.pad_token = torch.Tensor([tokenizer_src.token_to_id(['[PAD]'])], dtype = torch.int64)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        src_tgt = self.dataset[index]
        src_text = src_tgt['translation'][self.src_lang]
        tgt_text = src_tgt['translation'][self.tgt_lang]

        src_tokens = self.tokenizer_src(src_text).ids
        tgt_tokens = self.tokenizer_tgt(tgt_text).ids

        src_padding_tokens_len = self.seq_length - len(src_tokens) - 2 # SOS and EOS
        tgt_padding_tokens_len = self.seq_length - len(tgt_tokens) - 1 # only SOS

        if src_padding_tokens_len < 0 or tgt_padding_tokens_len < 0:
            raise ValueError('Sentence longer than seqennce length: token lenght exceeded by {src_padding_tokens_len}')
        
        # add SOS, EOS and PAD tokens to the encoder input 
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(src_tokens, dtype=torch.int64), # change this to tensor.clone as stated in the warning
                self.eos_token,
                torch.tensor(self.pad_token * src_padding_tokens_len, dtype=torch.int64)
            ]
        )

        # add SOS and PAD to the decoder input tokens
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(tgt_tokens, dtype=torch.int64), # change this to tensor.clone as stated in the warning
                torch.tensor(self.pad_token * tgt_padding_tokens_len, dtype=torch.int64)
            ]
        )

        # add only EOS and PAD. what we expect as the output from decoder
        label = torch.cat(
            [
                torch.tensor(decoder_input, dtype=torch.int64), # change this to tensor.clone as stated in the warning
                self.eos_token,
                torch.tensor(self.pad_token * tgt_padding_tokens_len, dtype=torch.int64)
            ]
        )

        return {
            'encoder_input': encoder_input, # (seq_length)
            'decoder_input': decoder_input, # (seq_length)
            'encoder_mask': (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),  # (1, 1, seq_length). actual input would be (batch, sentences, words)
            'decoder_mask': (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, 1, seq_length) & (1, seq_length, seq_length)
            'label': label,  # (seq_length)
            'src_text': src_text,
            'tgt_text': tgt_text
        }
    

def causal_mask(size):
    return torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int) == 0

