# Music and sound emotions

### Reproduce

#### Using `pdm`

This project was developed using pdm and intel MKL libraries. To setup the same
identical environment, do as follows:

1. Install pdm (e.g. using pipx)
2. Enter the project directory and run `pdm sync`
3. Download IADS-E and PMEmo datasets ad unzip them
4. Download OpenSmile 3.0.1
5. Download and extract the datasets each in a different directory (IADS-E, IADS-2, PMEmo)
6. Modify `music_sound_emotions/settings.py` to match your paths:

- the path to OpenSmile root directory
- the paths to the datasets root directories

8. From the project root run:

- `pdm features` to extract features
- `pdm experiment` to reproduce our experiments
- `pdm parse experiment*.log` to parse the logs
