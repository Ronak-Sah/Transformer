from src.components.prediction import PredictionPipeline

translator=PredictionPipeline()

out=translator.translate_sentence("help")

print(out)