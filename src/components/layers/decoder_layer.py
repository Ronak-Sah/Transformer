import torch
import torch.nn as nn


from src.components.attention.multi_head_attention import MultiHeadAttention
from src.components.attention.cross_attention import CrossMultiHeadAttention
from src.components.network.feed_forward import FeedForwardNetwork
class Decoder_Layer(nn.Module):
    def __init__(self,emb_dim,hidden_dim,num_heads,drop_prob=0.1):
        super().__init__()
        self.multi_attention=MultiHeadAttention(emb_dim,num_heads)
        self.norm1=nn.LayerNorm(emb_dim)
        self.dropout1=nn.Dropout(p=drop_prob)
        self.cross_attention=CrossMultiHeadAttention(emb_dim,num_heads)
        self.norm2=nn.LayerNorm(emb_dim)
        self.dropout2=nn.Dropout(p=drop_prob)
        self.ffn=FeedForwardNetwork(emb_dim,hidden_dim)
        self.norm3=nn.LayerNorm(emb_dim)
        self.dropout3=nn.Dropout(p=drop_prob)
        
    def forward(self,dec_in,enc_in ,self_attention_mask=None, cross_attention_mask=None):
        residual_x=dec_in
        x=self.multi_attention(dec_in, mask=self_attention_mask)
        x=self.dropout1(x)
        x=x+residual_x
        x=self.norm1(x)
        
        residual_x=x
        x=self.cross_attention(q=x,kv=enc_in,mask=cross_attention_mask)
        x=self.dropout2(x)
        x=x+residual_x
        x=self.norm2(x)

        residual_x=x
        x=self.ffn(x)
        x=self.dropout3(x)
        x=x+residual_x
        x=self.norm3(x)
        return x

