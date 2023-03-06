import subprocess
from pathlib import Path

from . import settings as S

if __name__ == "__main__":
    dirs = S.IADS_DIR + S.PMEMO_DIR
    for d in [S.IADSE_DIR]:
        d = Path(d)
        # convert mp3 to wav
        mp3files = d.glob("**/*.mp3")
        for mp3path in mp3files:
            if not mp3path.with_suffix(".wav").exists():
                subprocess.check_call(
                    [S.FFMPEG, "-i", mp3path, mp3path.with_suffix(".wav")]
                )
        # process wav files
        wavfiles = d.glob("**/*.wav")
        for wavpath in wavfiles:
            subprocess.check_call(
                [
                    S.OPEN_SMILE_EXE,
                    "-C",
                    S.OPEN_SMILE_CONFIG,
                    "-I",
                    wavpath,
                    "-csvoutput",
                    d / S.FEATURE_FILE,
                    "-instname",
                    wavpath.name,
                ]
            )
