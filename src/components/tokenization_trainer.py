import os
from src.logger import logger
from src.entity import TokenizationTrainerConfig
import pandas as pd
from bpetokenizer import BPETokenizer

class Tokenization_Trainer:
    def __init__(self,config: TokenizationTrainerConfig):

        self.special_tokens = {
        "<|endoftext|>": 1001,
        "<|startoftext|>": 1002,
        "[SPECIAL1]": 1003,
        "[SPECIAL2]": 1004,
        }
        self.config=config
        self.english_tokenizer = BPETokenizer(special_tokens=self.special_tokens)
        self.hindi_tokenizer = BPETokenizer(special_tokens=self.special_tokens)
        
        
    
    def tokenize(self):
        eng,hin=self.convert()

        eng_text = "\n".join(eng)
        hin_text = "\n".join(hin)

        self.english_tokenizer.train(eng_text, vocab_size=1000, verbose=True)
        self.hindi_tokenizer.train(hin_text, vocab_size=1000, verbose=True)

        os.makedirs(self.config.tokenizer_path, exist_ok=True)

        hin_path = os.path.join(self.config.tokenizer_path, "hin.json")
        eng_path = os.path.join(self.config.tokenizer_path, "eng.json")

        self.hindi_tokenizer.save(hin_path)
        self.english_tokenizer.save(eng_path)


    def convert(self):
        df =pd.read_csv(self.config.data_path)
        df=df.head(1000)
        english=[str(line) for line in df["English"]]
        hindi=[str(line) for line in df["Hindi"]]
        return english, hindi
        

