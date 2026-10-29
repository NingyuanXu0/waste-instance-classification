"""Shared paths and runtime settings for the classification entry points."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get('WASTE_DATA_ROOT', ROOT / 'data' / 'coco')).expanduser().resolve()
ANNOTATIONS = Path(os.environ.get('WASTE_ANNOTATIONS', DATA_ROOT / 'annotations.json')).expanduser().resolve()
OUTPUT_DIR = Path(os.environ.get('WASTE_OUTPUT_DIR', ROOT / 'outputs')).expanduser().resolve()
DEVICE = os.environ.get('WASTE_DEVICE', 'cpu')
BATCH_SIZE = int(os.environ.get('WASTE_BATCH_SIZE', '32'))
EPOCHS = int(os.environ.get('WASTE_EPOCHS', '1000'))
SPLIT_SEED = 42
