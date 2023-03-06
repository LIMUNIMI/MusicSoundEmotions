from pathlib import Path

#############################################
############# Configuration #################
#############################################

# modify these paths for your environment
OPEN_SMILE_DIR = Path("/opt/opensmile-3.0.1")
PMEMO_DIR = ["/datasets/emotions/PMEmo2019/"]
IADSE_DIR = "/datasets/emotions/IADS-E"
IADS2_DIR = "/datasets/emotions/IADS2007/"
FFMPEG = "/usr/bin/ffmpeg"

#############################################
########### End configuration ###############
#############################################

# Feature extraction
OPEN_SMILE_CONFIG = OPEN_SMILE_DIR / "config" / "is09-13" / "IS13_ComParE.conf"
if (OPEN_SMILE_DIR / "build.sh").exists():
    OPEN_SMILE_EXE = (
        OPEN_SMILE_DIR / "build" / "progsrc" / "smilextract" / "SMILExtract"
    )
else:
    OPEN_SMILE_EXE = OPEN_SMILE_DIR / "bin" / "SMILExtract"

# dataset paths
IADS_DIR = [IADS2_DIR, IADSE_DIR]
FEATURE_FILE = "static_features.csv"

# experiments
N_SPLITS = 5
RATIOS = 0.0, 0.25, 0.5, 0.75, 1.0
LABELS = "AroMN", "ValMN"

# logging


class Tlog:
    _log_spaces = 0

    def __call__(self, message: str = ""):
        print(" " * self._log_spaces + message)


tlog = Tlog()
