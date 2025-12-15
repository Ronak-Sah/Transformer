import torch.nn as nn
import torch
from bpetokenizer import BPETokenizer
from src.components.network.position_encoding import PositionalEncoding
from torch.nn.utils.rnn import pad_sequence

eng_tokenizer = BPETokenizer()
eng_tokenizer.load(r"artifacts\\tokenization_trainer\\tokenizer\\eng.json", mode="json")

hin_tokenizer=BPETokenizer()
hin_tokenizer.load(r"artifacts\\tokenization_trainer\\tokenizer\\hin.json", mode="json")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Eng_Sentence_Embedding(nn.Module):

    def __init__(self, max_sequence_length, d_model):
        super().__init__()
        self.vocab_size = len(eng_tokenizer.vocab)
        self.max_sequence_length = max_sequence_length
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.position_encoder = PositionalEncoding(d_model, self.max_sequence_length)
        self.dropout = nn.Dropout(p=0.1)


    
    
    def forward(self, x, start_token, end_token): # sentence
        # x = self.batch_tokenize(x, start_token, end_token)
        
        x = self.embedding(x)

        seq_len = x.size(1)

        pos = self.position_encoder()[:, :seq_len, :]  # [1, S, D]
        pos = pos.expand(x.size(0), -1, -1).to(x.device)

        # print("x:", x.shape)
        # print("pos:", pos.shape)

        # if seq_len > self.max_sequence_length:
        #     raise ValueError(f"Sequence length {seq_len} exceeds max_sequence_length {self.max_sequence_length}")

        x = self.dropout(x + pos)
        return x
    


class Hin_Sentence_Embedding(nn.Module):

    def __init__(self, max_sequence_length, d_model):
        super().__init__()
        self.vocab_size = len(hin_tokenizer.vocab)
        self.max_sequence_length = max_sequence_length
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.position_encoder = PositionalEncoding(d_model, max_sequence_length)
        self.dropout = nn.Dropout(p=0.1)


    
    
    def forward(self, x, start_token, end_token): # sentence
        
        x = self.embedding(x)

        seq_len = x.size(1)

        pos = self.position_encoder()[:, :seq_len, :]  # [1, S, D]
        pos = pos.expand(x.size(0), -1, -1).to(x.device)

        # print("x:", x.shape)
        # print("pos:", pos.shape)


        x = self.dropout(x + pos)
        return x
    

class Tokenizer():
    def __init__(self,batch, start_token="<start>", end_token="<end>"):
        self.batch=batch
        self.start_token=start_token
        self.end_token=end_token
    def hin_batch_tokenize(self):
        tokenized = []

        for sentence in self.batch:
            text = f"{self.start_token} {sentence} {self.end_token}"

            ids = hin_tokenizer.encode(text)

            tokenized.append(torch.tensor(ids, dtype=torch.long))

        tokenized_padded = pad_sequence(tokenized, batch_first=True, padding_value=0)
        return tokenized_padded
    

    def eng_batch_tokenize(self):
        tokenized = []

        for sentence in self.batch:
            text = f"{self.start_token} {sentence} {self.end_token}"

            ids = eng_tokenizer.encode(text)

            tokenized.append(torch.tensor(ids, dtype=torch.long))

        tokenized_padded = pad_sequence(tokenized, batch_first=True, padding_value=0)
        return tokenized_padded