import torch.nn as nn
from src.components.models.encoder import Encoder
from src.components.models.decoder import Decoder
import torch

class Transformer(nn.Module):
    def __init__(self, emb_dim, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, hin_vocab_size):
        super().__init__()
        self.encoder = Encoder(emb_dim, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length)
        self.decoder = Decoder(emb_dim, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length)
        self.linear = nn.Linear(emb_dim, hin_vocab_size)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self, dec_in, enc_in, encoder_self_attention_mask=None, 
            decoder_self_attention_mask=None, decoder_cross_attention_mask=None, start_token=None, end_token=None): 
        x = self.encoder(enc_in, encoder_self_attention_mask, start_token, end_token)
        out = self.decoder(dec_in, x, decoder_self_attention_mask, decoder_cross_attention_mask, start_token, end_token)
        out = self.linear(out)
        return out