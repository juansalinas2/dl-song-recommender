from pathlib import Path

# project root
ROOT = Path(__file__).resolve().parents[2]

# artifacts directory (gitignored)
ARTIFACTS_DIR = ROOT / 'artifacts'

# audio directories
AUDIO_DIR = ARTIFACTS_DIR / 'audio'
STEMS_DIR = ARTIFACTS_DIR / 'stems'

# spectrogram directories
SPECTROGRAM_PNG_DIR = ARTIFACTS_DIR / 'spectrograms_png'
SPECTROGRAM_RAW_DIR = ARTIFACTS_DIR / 'spectrograms_raw'

# metadata directories
DATA_DIR = ROOT / 'data'

# configs
CONFIGS_DIR = ROOT / 'configs'

# models
# tbd...
