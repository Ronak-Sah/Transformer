from src.components.prediction import PredictionPipeline

translator=PredictionPipeline()

out=translator.translate_sentence("bihar is a state of india")

print(out)