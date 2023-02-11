from pathlib import Path

# Feature extraction
OPEN_SMILE_DIR = Path("/opt/opensmile-3.0.1-linux-x64")
OPEN_SMILE_CONFIG = OPEN_SMILE_DIR / "config/is09-13/IS13_ComParE.conf"
OPEN_SMILE_EXE = OPEN_SMILE_DIR / "bin/SMILExtract"
FFMPEG = "/usr/bin/ffmpeg"

# dataset paths
PMEMO_DIR = ["/datasets/emotions/PMEmo2019/"]
IADSE_DIR = "/datasets/emotions/IADS-E/"
IADS2_DIR = "/datasets/emotions/IADS-2/"
IADS_DIR = [IADS2_DIR, IADSE_DIR]
FEATURE_FILE = "static_features.csv"
