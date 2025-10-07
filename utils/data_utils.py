import re
import pandas as pd
import os

def get_artist_from_id(id_str):
    """Extract and format artist name from id string."""
    artist = id_str.split('_')[0]
    artist = re.sub(r'(?<!^)([A-Z])', r' \1', artist)
    return artist

def ensure_output_dir(path="outputs"):
    """Create output directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def load_phrasebur_csv(filename="data/phrasebur_filtered.csv"):
    """Load filtered PhraseBur data with correct separator."""
    return pd.read_csv(filename, sep=';')
