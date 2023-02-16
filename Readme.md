# Music and sound emotions

### Reproduce

1. Install python 3.9 (e.g. using conda or pyenv)
2. Install pdm (e.g. using pipx)
3. Enter the project directory and run `pdm sync`
4. Download IADS-E and PMEmo datasets ad unzip them
5. Download OpenSmile 3.0.1
6. Modify `music_sound_emotions/settings.py` to match your paths:
  1. the path to OpenSmile root directory
  2. the paths to the datasets root directories
7. From the project root, run `pdm experiment` to reproduce our experiments
