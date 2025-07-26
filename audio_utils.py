import librosa
import numpy as np


class AudioContainer:
    """This class is used to hold utility functions and attributes for audio processing"""

    def __init__(self, file_path, chunk_secs=3):
        """Loads file and initializes waveform (y) and sampling rate (sr)"""

        self.file_path = file_path
        self.chunk_secs = chunk_secs

        y, sr = librosa.load(self.file_path, sr=None)
        self.y = y
        self.sr = sr

    def get_spec_chunks(self):
        """Divides audio into chunks with specified length
        Returns chunks, chunk shape"""

        y, sr = self.y, self.sr
        hop_length = 512

        # Convert to Mel-spectrogram

        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length)
        mel_spec = librosa.power_to_db(mel_spec)

        # Extend to 3d for Keras

        mel_spec = np.expand_dims(mel_spec, axis=2)

        # Divide into 10-sec chunks

        frame_len = sr / hop_length * self.chunk_secs

        chunks = [
            mel_spec[:, int(i) : int(i) + int(frame_len)]
            for i in np.arange(0.0, mel_spec.shape[1], frame_len)
        ][:-1]  # Drop last chunk to ensure dimension regularity

        return np.array(chunks)


"""
# Testing
#
audio_utils = AudioContainer("assets/examples/music_net/1728.wav")
chunks, chunk_shape = audio_utils.get_spec_chunks()
print(len(chunks), chunk_shape)
"""
