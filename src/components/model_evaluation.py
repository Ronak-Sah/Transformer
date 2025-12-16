import os
from src.logger import logger
from src.entity import ModelEvaluationConfig,ModelTrainerConfig
import torch
import sacrebleu
import torch.nn as nn
from torch.utils.data import  DataLoader
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
import pandas as pd
from src.components.models.transformer import Transformer
from bpetokenizer import BPETokenizer
from src.components.model_trainer import TranslationDataset
from src.components.network.embeddings import Tokenizer


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



import torch

def greedy_decode(model, enc_tokens, max_len, device):
    model.eval()

    start_id = hin_tokenizer.special_tokens["<start>"]
    end_id   = hin_tokenizer.special_tokens["<end>"]

    ys = torch.tensor([[start_id]], device=device)

    for _ in range(max_len):
        dec_self_mask = create_decoder_self_attention_mask(ys)

        with torch.no_grad():
            out = model(
                enc_in=enc_tokens,
                dec_in=ys,
                decoder_self_attention_mask=dec_self_mask,
                decoder_cross_attention_mask=None
            )

        next_token = out[:, -1].argmax(dim=-1, keepdim=True)
        ys = torch.cat([ys, next_token], dim=1)

        if next_token.item() == end_id:
            break

    return ys




class Model_Evaluation:
    def __init__(self,train_config:ModelTrainerConfig,config: ModelEvaluationConfig,):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.train_config = train_config
        self.model = Transformer(
            emb_dim=self.train_config.emb_dim,
            ffn_hidden=self.train_config.ffn_hidden,
            num_heads=self.train_config.num_heads,
            drop_prob=self.train_config.drop_prob,
            num_layers=self.train_config.num_layers,
            max_sequence_length=self.train_config.max_sequence_length,
            hin_vocab_size=hin_vocab_size
        ).to(self.device)

        model_path = config.model_path
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))


        logger.info("Model loaded succesfully for evaluation")

    
    def evaluate(self):
        
        self.model.eval()
        loss_fn = nn.CrossEntropyLoss(ignore_index=0)

        total_loss = 0
        references = []
        hypotheses = []

        data_path=self.config.data_path
        df =pd.read_csv(data_path)
        df=df.iloc[5000:5501]
        english=[str(line) for line in df["English"]]
        hindi=[str(line) for line in df["Hindi"]]
        

        dataset = TranslationDataset(english, hindi)

        dataloader = DataLoader(
            dataset,
            batch_size=8,  
            shuffle=True
        )

        with torch.no_grad():
            for eng_batch, hin_batch in dataloader:

                eng_tok = Tokenizer(eng_batch)
                hin_tok = Tokenizer(hin_batch)

                enc_tokens = eng_tok.eng_batch_tokenize().to(self.device)
                tgt_tokens = hin_tok.hin_batch_tokenize().to(self.device)

                dec_self_mask = create_decoder_self_attention_mask(tgt_tokens)

                logits = self.model(
                    enc_in=enc_tokens,
                    dec_in=tgt_tokens,
                    decoder_self_attention_mask=dec_self_mask,
                    decoder_cross_attention_mask=None
                )
                loss = loss_fn(
                    logits.view(-1, logits.size(-1)),
                    tgt_tokens.view(-1)
                )

                total_loss += loss.item()

        batch_size=enc_tokens.size(0)
        for i in range(batch_size):
            pred_ids = greedy_decode(self.model, enc_tokens[i:i+1], max_len=100,device=self.device)

            pred_text = hin_tokenizer.decode(pred_ids.squeeze().tolist())
            ref_text  = hin_batch[i]

            hypotheses.append(pred_text.split())
            references.append([ref_text.split()])

        
        bleu=corpus_bleu(references, hypotheses)
        score = sentence_bleu(references, hypotheses)
        print("sentence_bleu score :",score)
     
        print("bleu :",bleu)
        print("Loss/len :",total_loss / len(dataloader))


        dict = {"bleu_score": bleu ,"loss":total_loss / len(dataloader)}

        df=pd.DataFrame(dict,index=["tranformer"])

        df.to_csv(self.config.metric_file_name,index=False)
        logger.info(f"Metric CSV creted at {self.config.metric_file_name}")

        return total_loss / len(dataloader), bleu
    

    