from src.config.configuration import ConfigurationManager
from src.components.tokenization_trainer import Tokenization_Trainer
from src.logger import logger


class Tokenization_Trainer_pipeline:
    def __init__(self):
        pass
    
    def main(self):
        config=ConfigurationManager()
        tokenization_trainer_config=config.get_tokenization_trainer()

        tokenization_trainer=Tokenization_Trainer(tokenization_trainer_config)
        tokenization_trainer.tokenize()
        
        