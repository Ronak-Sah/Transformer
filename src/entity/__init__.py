# It contains dataclasses that define the structure of configuration objects.

from dataclasses import dataclass
from pathlib import Path

# Configuration for data ingestion

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir:Path
    source_url :str
    local_data_file :Path
    unzip_dir :Path




@dataclass(frozen=True)
class TokenizationTrainerConfig:
    root_dir : Path
    data_path : str
    tokenizer_path : str


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir : Path
    data_path : str
    tokenizer_path : str
    epochs: 1
    emb_dim: int
    ffn_hidden: int
    num_heads: int
    drop_prob: float
    num_layers: int
    max_sequence_length : int
    batch_size : int


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir : Path
    data_path : Path
    model_path : Path
    tokenizer_path : Path
    metric_file_name : Path
    

