from src.config.configuration import ConfigurationManager
from src.components.model_evaluation import Model_Evaluation 
from src.logger import logger


class Model_Evaluation_pipeline:
    def __init__(self):
        pass
    
    def main(self):
        config=ConfigurationManager()
        model_trainer_config=config.get_model_trainer()
        model_evaluate_config=config.get_model_evaluation()
        model_evaluater=Model_Evaluation(model_trainer_config,model_evaluate_config)
        model_evaluater.evaluate()
        