import os
from src.logger import logger
from src.entity import ModelTrainerConfig
import torch
import torch.nn as nn
from src.components.models.transformer import Transformer
from src.components.network.embeddings import Tokenizer
from bpetokenizer import BPETokenizer
import pandas as pd

hin_tokenizer=BPETokenizer()
hin_tokenizer.load(r"artifacts\\tokenization_trainer\\tokenizer\\hin.json", mode="json")
hin_vocab_size=len(hin_tokenizer.vocab)



def create_decoder_self_attention_mask(tokens, pad_id=0):
    """
    tokens: [B, T_dec]
    returns: [B, 1, T_dec, T_dec]
    """
    B, T = tokens.shape

    pad_mask = (tokens != pad_id).unsqueeze(1).unsqueeze(2)  # [B,1,1,T]
    causal_mask = torch.tril(torch.ones(T, T, device=tokens.device))  # [T,T]

    return pad_mask * causal_mask


def create_cross_attention_mask(dec_tokens, enc_tokens, pad_id=0):
    """
    dec_tokens: [B, T_dec]
    enc_tokens: [B, T_enc]
    returns: [B, 1, T_dec, T_enc]
    """
    B, T_dec = dec_tokens.shape
    _, T_enc = enc_tokens.shape

    enc_pad_mask = (enc_tokens != pad_id).unsqueeze(1).unsqueeze(2)  # [B,1,1,T_enc]
    return enc_pad_mask.expand(B, 1, T_dec, T_enc)


class Model_Trainer:
    def __init__(self,config: ModelTrainerConfig):
        self.config= config
        self.device= "cuda" if torch.cuda.is_available() else "cpu"

        


    def train(self):
        data_path=self.config.data_path
        df =pd.read_csv(data_path)
        df=df.head(100)
        english=[str(line) for line in df["English"]]
        hindi=[str(line) for line in df["Hindi"]]
        transformer=Transformer(emb_dim=self.config.emb_dim, ffn_hidden=self.config.ffn_hidden,
                num_heads=self.config.num_heads, drop_prob=self.config.drop_prob, 
                num_layers=self.config.num_layers, max_sequence_length=self.config.max_sequence_length, hin_vocab_size=hin_vocab_size).to(self.device)
        
        optimizer=torch.optim.SGD(transformer.parameters(),lr=0.01)
        loss_fn=nn.CrossEntropyLoss(ignore_index=0)

        eng_tokenizer=Tokenizer(english)
        hin_tokenizer=Tokenizer(hindi)
        english_tokens = eng_tokenizer.eng_batch_tokenize().to(self.device)
        hindi_tokens = hin_tokenizer.hin_batch_tokenize().to(self.device)

        dec_self_mask = create_decoder_self_attention_mask(hindi_tokens)
        cross_mask = create_cross_attention_mask(hindi_tokens, english_tokens)

        for epoch in range(self.config.epochs):
            transformer.train()
            optimizer.zero_grad()

            y_pred=transformer(enc_in=english_tokens,dec_in=hindi_tokens,decoder_self_attention_mask= dec_self_mask,decoder_cross_attention_mask=None)

            loss = loss_fn(y_pred.view(-1, y_pred.size(-1)), hindi_tokens.view(-1))
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch+1}/{self.config.epochs}, Loss: {loss.item()}")





        


    