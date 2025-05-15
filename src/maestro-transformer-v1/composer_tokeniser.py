import pandas as pd
from huggingface_hub import hf_hub_download
import json
import os
from pathlib import Path

from src import PROCESSED_DATA_DIR

# Define constants
DATASET_REPO_ID = "ddPn08/maestro-v3.0.0"
METADATA_FILENAME = "maestro-v3.0.0.csv"

# Determine project root assuming the script is in src/maestro-tranformer/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_VOCAB_FILENAME = "composer_vocab.json"

def download_metadata(repo_id: str, filename: str) -> pd.DataFrame:
    """Downloads the metadata CSV from Hugging Face and loads it into a pandas DataFrame."""
    print(f"Downloading {filename} from {repo_id}...")
    try:
        metadata_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset"
        )
        print(f"Successfully downloaded to {metadata_path}")
        df = pd.read_csv(metadata_path)
        print("Successfully loaded CSV into DataFrame.")
        return df
    except Exception as e:
        print(f"Error downloading or parsing metadata CSV: {e}")
        raise

def create_composer_vocabulary(df: pd.DataFrame) -> dict:
    """Extracts unique composers and creates a token vocabulary (name to ID)."""
    if 'canonical_composer' not in df.columns:
        print("Error: 'canonical_composer' column not found in the DataFrame.")
        # Fallback or alternative column if needed, for now, raise error
        raise ValueError("'canonical_composer' column is missing.")
    
    unique_composers = sorted(list(df['canonical_composer'].unique()))
    composer_to_id = {composer: i for i, composer in enumerate(unique_composers)}
    print(f"Created vocabulary with {len(composer_to_id)} unique composers.")
    return composer_to_id

def save_vocabulary(vocabulary: dict, output_dir: Path, filename: str):
    """Saves the vocabulary to a JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    with open(output_path, 'w') as f:
        json.dump(vocabulary, f, indent=4)
    print(f"Composer vocabulary saved to {output_path}")

def main():
    """Main function to orchestrate the tokenization and saving process."""
    print("Starting composer tokenization process...")
    try:
        metadata_df = download_metadata(DATASET_REPO_ID, METADATA_FILENAME)
        composer_vocab = create_composer_vocabulary(metadata_df)
        save_vocabulary(composer_vocab, PROCESSED_DATA_DIR, OUTPUT_VOCAB_FILENAME)
        print("Composer tokenization process completed successfully.")
    except Exception as e:
        print(f"An error occurred during the process: {e}")

if __name__ == "__main__":
    main()
