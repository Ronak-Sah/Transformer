import os
from src.logger import logger
from src.entity import ModelTrainerConfig
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from src.components.models.transformer import Transformer
from src.components.network.embeddings import Tokenizer
from bpetokenizer import BPETokenizer
import pandas as pd

hin_tokenizer=BPETokenizer()
hin_tokenizer.load(r"artifacts\\tokenization_trainer\\tokenizer\\hin.json", mode="json")
hin_vocab_size=len(hin_tokenizer.vocab)




class TranslationDataset(Dataset):
    def __init__(self, english, hindi):
        self.english = english
        self.hindi = hindi

    def __len__(self):
        return len(self.english)

    def __getitem__(self, idx):
        return self.english[idx], self.hindi[idx]


def create_decoder_self_attention_mask(tokens, pad_id=0):
    """
    tokens: [B, T_dec]
    returns: [B, 1, T_dec, T_dec]
    """
    B, T = tokens.shape

    pad_mask = (tokens != pad_id).unsqueeze(1).unsqueeze(2)  # [B,1,1,T]
    causal_mask = torch.tril(torch.ones(T, T, device=tokens.device))  # [T,T]

    return pad_mask * causal_mask





class Model_Trainer:
    def __init__(self,config: ModelTrainerConfig):
        self.config= config
        self.device= "cuda" if torch.cuda.is_available() else "cpu"

        self.transformer = Transformer(
            emb_dim=self.config.emb_dim,
            ffn_hidden=self.config.ffn_hidden,
            num_heads=self.config.num_heads,
            drop_prob=self.config.drop_prob,
            num_layers=self.config.num_layers,
            max_sequence_length=self.config.max_sequence_length,
            hin_vocab_size=hin_vocab_size
        ).to(self.device)

        model_path = os.path.join(config.root_dir, "transformer_model.pth")
        if os.path.exists(model_path):
           
            self.transformer.load_state_dict(torch.load(model_path, map_location=self.device))
            

    def train(self):
        data_path=self.config.data_path
        df =pd.read_csv(data_path)
        df=df.head(5000)
        english=[str(line) for line in df["english_sentence"]]
        hindi=[str(line) for line in df["hindi_sentence"]]
        
        optimizer = torch.optim.Adam(self.transformer.parameters(),lr=1e-4,betas=(0.9, 0.98),eps=1e-9)

        loss_fn=nn.CrossEntropyLoss(ignore_index=0)

        

        dataset = TranslationDataset(english, hindi)

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,  
            shuffle=True
        )


        for epoch in range(self.config.epochs):
            self.transformer.train()
            total_loss = 0.0

            for eng_batch, hin_batch in dataloader:
                eng_tok = Tokenizer(eng_batch)
                hin_tok = Tokenizer(hin_batch)

                english_tokens = eng_tok.eng_batch_tokenize().to(self.device)
                hindi_tokens   = hin_tok.hin_batch_tokenize().to(self.device)

                dec_in  = hindi_tokens[:, :-1]
                dec_out = hindi_tokens[:, 1:]

                dec_self_mask = create_decoder_self_attention_mask(dec_in)
   
                # enc_in: (batch, src_len, d_model)
                # cross_mask = (enc_in.sum(dim=-1) != 0).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, src_len)

                optimizer.zero_grad()
                y_pred=self.transformer(enc_in=english_tokens,dec_in=dec_in,decoder_self_attention_mask= dec_self_mask,decoder_cross_attention_mask=None)


                loss = loss_fn(y_pred.reshape(-1, y_pred.size(-1)),dec_out.reshape(-1))
            
                loss.backward()
                optimizer.step()

                total_loss += loss.item()


            print(f"Epoch {epoch+1}/{self.config.epochs}, Loss: {total_loss: .4f}")




        os.makedirs(self.config.root_dir, exist_ok=True)

        model_save_path = os.path.join(self.config.root_dir, "transformer_model.pth")

        torch.save(self.transformer.state_dict(), model_save_path)

        logger.info(f"Model saved at: {model_save_path}")





        


    