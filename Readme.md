# Music and sound emotions

### Reproduce

#### Using `pdm`
This project was developed using pdm and intel MKL libraries. To setup the same
identical environment, do as follows:

1. Install python 3.9 (e.g. using conda or pyenv)
2. Install pdm (e.g. using pipx)
3. Enter the project directory and run `pdm sync --no-self`
4. Download IADS-E and PMEmo datasets ad unzip them
5. Download OpenSmile 3.0.1
6. Modify `music_sound_emotions/settings.py` to match your paths:
  * the path to OpenSmile root directory
  * the paths to the datasets root directories
7. From the project root run:
  * `pdm features` to extract features
  * `pdm experiment` to reproduce our experiments

#### Using `API`

You can also use the API to perform the experiments with different python versions
(e.g. using conda/mamba with Python 3.10, or without using MKL libraries).

For this approach, see below.

### API

To use this code into your own, just install it with pip:
  `pip install git+https://github.com/LIMUNIMI/MusicSoundEmotions.git`

Youcan perform feature extraction using `python -m music_sound_emotions.features`.

You can perform the experiments using `python -m music_sound_emotions.experiments`.

You can run one single experiment using `python -m
music_sound_emotions.validation tune_and_validate --label=<LABEL> --p=<P>`, where:
  * `P` must be float between 0 and 1 and corresponds to how much of IADS is mixed
    with PMEmo
  * `LABEL` must be one of `AroMN`, `ValMN`, `AroSD`, `ValSD` and correspond to
    Arousal/Valence mean/standard deviation.
