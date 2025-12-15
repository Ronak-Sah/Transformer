import torch.nn as nn
from src.components.layers.encoder_layer import Encoder_Layer
from src.components.network.embeddings import Eng_Sentence_Embedding


class Sequential_Encoder(nn.Module):
    def __init__(self, emb_dim, num_heads, ffn_hidden, drop_prob, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([Encoder_Layer(emb_dim, num_heads,ffn_hidden, drop_prob)
                                     for _ in range(num_layers)])

    def forward(self, x, self_attention_mask=None):
        for layer in self.layers:
            x = layer(x, self_attention_mask)
        return x
    


class Encoder(nn.Module):
    def __init__(self, emb_dim, ffn_hidden, num_heads, drop_prob, num_layers,max_sequence_length):
        super().__init__()
        self.sentence_embedding = Eng_Sentence_Embedding(max_sequence_length, emb_dim)
        self.layers = Sequential_Encoder(emb_dim, num_heads, ffn_hidden, drop_prob, num_layers)

    def forward(self, x, self_attention_mask, start_token, end_token):
        x=x.long()
        x = self.sentence_embedding(x, start_token, end_token)
        x = self.layers(x, self_attention_mask)
        return x