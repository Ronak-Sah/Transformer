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


# Configuration for data tranformation

@dataclass(frozen=True)
class TokenizationTrainerConfig:
    root_dir : Path
    data_path : str
    tokenizer_path : str

