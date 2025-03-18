import math
import torch
from torch import nn

# Input embeddings

class WordEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super(WordEmbedding).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, self.d_model) # (number of embeddings, embedding size)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model) # they divide by the sqrt of the embedding size in the paper


# positional encoding

class PositionalEncoding(nn.Module):
    def __init__(self, seq: int, d_model: int, dropout: float = 0.1):
        super(PositionalEncoding).__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq, d_model) # (seq, d_model)
        position = torch.arange(0, seq, dtype=torch.float).unsqueeze(1) # (seq, 1)
        # div_term = 1/torch.pow(10000, torch.arange(0, d_model, 2)/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)) # (seq,)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, seq, d_model)

        # when you want to store a tensor but not as a model parameter, use buffer
        torch.register_buffer('pe', pe)

    def forward(self, x):
        # add the positional encodings. limit the seq as it could be different based on the number of words (1, seq, d_model)
        x = x + self.pe[:, :x.shape[1], :].requires_grad_(False)
        return self.dropout(x)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super(FeedForward).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        # (N, seq, d_model) -> (N, seq, d_dff) -> (N, seq, d_model)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.h = h
        self.d = d_model // h

        assert(d_model % h == 0), "Embedding space must be divisible by number of heads"

        self.w_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_v = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_0 = nn.Linear(self.d * self.h, self.d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask):
        # Batch size and number of words or sequence
        N, seqq, seqk, seqv = N = q.shape[0] , q.shape[1], k.shape[1], v.shape[1]

        query = self.w_q(q) # (batch, seq, e) -> (batch, seqq, e)
        key = self.w_k(k) # (batch, seq, e) -> (batch, seqk, e)
        value = self.w_v(v) # (batch, seq, e) -> (batch, seqv, e)
        # although I name seqq, seqk and seqv separately, it should be noted that seqk and seqv are always the same

        # splitting the embedding dimension
        # reshape the matrix and transpose its 2nd and 3rd dimension (seq and heads), so each head sees all the words but only a part of the embedding space
        # you can imagine this in 3d how in the start, each batch contains matrices with all the words divided into smaller sequences of hxd
        # after transpose, each batch contains all the heads containing nth "smaller sequence" (or part of the embedding)
        query = query.view(N, seqq, self.h, self.d).transpose(1, 2) # (batch, seqq, e) -> (batch, seqq, h, e) -> (batch, h, seqq, e)
        key = key.view(N, seqk, self.h, self.d).transpose(1, 2) # (batch, seqk, e) -> (batch, seqk, h, e) -> (batch, h, seqk, e)
        value = value.view(N, seqv, self.h, self.d).transpose(1, 2) # (batch, seqv, e) -> (batch, seqv, h, e) -> (batch, h, seqv, e)
        # or should I just do q.view(N, h, seq, d)? maybe experiment with einsum and the matmul operators to see if it makes any diff

        # now we need to multiply Q with K (transpose) to calculate attention scores
        # (N, h, seqq, d) x (N, h, d, seqk) -> (N, h, seqq, seqk)
        self.attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d) # calculates attention by multiplying each head in the ket and query space

        # apply mask and rop out
        if mask:
            self.attention_scores = self.attention_scores.masked_fill(mask == 0, float("-inf"))

        self.attention_scores = torch.softmax(self.attention_scores, dim=-1) # attention on key seq dimension

        if self.dropout:
            self.attention_scores = self.dropout(self.attention_scores)

        # since seqk and seqv are always same, this doesnt cause any shape error
        # (N, h, seqq, seqk) -> (N, h, seqv, d) -> (N, seqq, e)
        out = torch.matmul(self.attention_scores, value).transpose(1, 2).contiguous().view(N, seqq, self.h * self.d)

        # (N, seqq, e) -> (N, seqq, e)
        return self.w_0(out)


# not using this layer but this was an interestsing implementation too
class ResidualConnection(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model) # when the 'normalized_shape' parameter is of type 'int', it requires the input to have its last dimension to be of that size

    def forward(self, x, sublayer):
        # you can apply norm before or after the sublayer computations, its a preference i guess
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, self_attention_block: MultiHeadAttention, feedforward_block: FeedForward, dropout: float = 0.1):
        super(EncoderBlock).__init__()
        self.self_attention_block = self_attention_block
        self.feedforward_block = feedforward_block

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, src_mask):
        # the original shape is maintained through all the layers so there wouldnt be any change in the shape
        # here we make skip connections similar to what happens in ResNet
        x_ = self.norm1(x)
        x_ = self.self_attention_block(x_, x_, x_, src_mask)
        x_ = self.dropout1(x_)
        x = x + x_

        x_ = self.norm2(x)
        x_ = self.feedforward_block(x_)
        x_ = self.dropout2(x_)
        x = x + x_

        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feedforward_block: FeedForward, dropout: float = 0.1):
        super(DecoderBlock).__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feedforward_block = feedforward_block

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # the original shape is maintained through all the layers so there wouldnt be any change in the shape
        x_ = self.norm1(x)
        x_ = self.self_attention_block(x_, x_, x_, src_mask)
        x_ = self.dropout1(x_)
        x = x + x_

        x_ = self.norm2(x)
        x_ = self.cross_attention_block(x_, encoder_output, encoder_output, tgt_mask)
        x_ = self.dropout2(x_)
        x = x + x_

        x_ = self.norm3(x)
        x_ = self.feedforward_block(x_)
        x_ = self.dropout3(x_)
        x = x + x_

        return x


class Encoder(nn.Module):
    def __init__(self, d_model: int, n: int, self_attention_block: MultiHeadAttention, feedforward_block: FeedForward, dropout: float = 0.1):
        super(Encoder).__init__()
        self.encoder_blocks = nn.ModuleList([EncoderBlock(d_model, self_attention_block, feedforward_block, dropout) for _ in range(n)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, src_mask):
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, src_mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, d_model: int, n: int, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feedforward_block: FeedForward, dropout: float = 0.1):
        super(Decoder).__init__()
        self.decoder_blocks = nn.ModuleList([DecoderBlock(d_model, self_attention_block, cross_attention_block, feedforward_block, dropout) for _ in range(n)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super(ProjectionLayer).__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (N, seq, d_model) -> (N, seq, vocab_size)
        return torch.log_softmax(self.proj(x), dim=-1) 


class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: WordEmbedding, tgt_embed: WordEmbedding, pos: PositionalEncoding, projection_layer: ProjectionLayer):
        super(Transformer).__init__()
        self.encoder = encoder
        self.decoer = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.pos = pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.pos(src)
        return self.encoder(src, src_mask)

    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, n: int = 6, d_model: int = 512, h: int = 8, d_ff: int = 2048, dropout: float = 0.1):
    # embedding for source and target
    src_embed = WordEmbedding(d_model, src_vocab_size)
    tgt_embed = WordEmbedding(d_model, tgt_vocab_size)

    # positional encoding 
    pos = PositionalEncoding(d_model, dropout)

    # sub layers for encoder
    encoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
    encoder_feedforward_block = FeedForward(d_model, d_ff, dropout)

    encoder = Encoder(d_model, n, encoder_self_attention_block, encoder_feedforward_block, dropout)

    # sub layers for decoder
    decoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
    decoder_cross_attention_block = MultiHeadAttention(d_model, h, dropout)
    decoder_feedforward_block = FeedForward(d_model, d_ff, dropout)

    decoder = Decoder(d_model, n, decoder_self_attention_block, decoder_cross_attention_block, decoder_feedforward_block, dropout)

    # projection layer to get the output in tgt vocab size
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, pos, projection_layer)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer