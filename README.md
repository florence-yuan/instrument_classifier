# Time-based multi-label instrument classification model (WIP)

CNN model trained on classical music audios from MusicNet (currently only trained on a small subset with representative instruments: 'Acoustic Grand Piano', 'Violin', 'Viola', 'Cello', 'Contrabass', 'French Horn', 'Oboe', 'Bassoon', 'Clarinet', 'Flute').  Instrument representation is compliant with MIDI patch file.

main.py (WIP): predict instruments present in 3-second segments; also predicts instruments present in entire audio

modeling.py: builds and trains CNN model

audio_utils.py: utilities for audio conversion to spectrograms

csv_utils.py: utilities for csv manipulation

config_utils: maps instrument indexes to labels (MIDI-compliant)

Dataset for training (.wav + .csv):
- 1728 Schubert Piano Quintet in A major
- 1872 Mozart Piano Trio No 3 in B-flat major
- 1916 Dvorak String Quartet No 10 in E-flat major
- 2075 Cambini Wind Quintet No 1 in B-flat Major
- 2127 Brahms Serenade No 1 in D Major (Wind and Strings Octet)
- 2313 Beethoven String Quartet No 15 in A minor

(from [the MusicNet dataset](https://www.kaggle.com/datasets/imsparsh/musicnet-dataset/data))
