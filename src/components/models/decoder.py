import torch.nn as nn
from src.components.layers.decoder_layer import Decoder_Layer
from src.components.network.embeddings import Hin_Sentence_Embedding



class SequentialDecoder(nn.Module):
    def __init__(self, emb_dim, ffn_hidden, num_heads, drop_prob, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                Decoder_Layer(emb_dim, ffn_hidden, num_heads, drop_prob)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, y, self_attention_mask=None, cross_attention_mask=None):
        for layer in self.layers:
            y = layer(x, y, self_attention_mask=self_attention_mask, cross_attention_mask=cross_attention_mask)
        return y


class Decoder(nn.Module):
    def __init__(self, emb_dim, ffn_hidden, num_heads, drop_prob, num_layers,max_sequence_length):
        super().__init__()
        self.sentence_embedding = Hin_Sentence_Embedding(max_sequence_length, emb_dim)
        self.layers =SequentialDecoder(emb_dim, ffn_hidden, num_heads, drop_prob, num_layers) 

    def forward(self, x, y, self_attention_mask, cross_attention_mask, start_token, end_token):
        x = x.long()

       

        x = self.sentence_embedding(x, start_token, end_token)
        y = self.layers(x, y, self_attention_mask= self_attention_mask, cross_attention_mask= cross_attention_mask)
        return y