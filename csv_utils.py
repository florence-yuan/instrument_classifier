# from librosa.core import audio
import pandas as pd
import numpy as np


class CsvContainer:
    def __init__(
        self,
        file_path,  # Path to CSV file
        chunk_len,  # Length of each segment = sampling_rate * segment_seconds
        audio_len,
        data_features=["start_time", "end_time", "instrument"],
        ins_dur_threshold=0.5,  # Threshold duration (0, 1) for instrument inclusion
    ):
        self.file_path = file_path
        self.chunk_len = chunk_len
        self.audio_len = audio_len
        self.ins_dur_threshold = ins_dur_threshold
        self.data = pd.read_csv(file_path)[data_features]
        self.data = self.data.dropna(axis=1)
        self.data = self.data.drop_duplicates()
        self.data = self.data.sort_values(by=["start_time", "end_time"])

    def describe(self):
        print(self.data.head(n=10))

        """
    def encode_ins(self):
        Apply label encoding to instruments
        Transform instruments to 0...n-1 (where n is the number of unique instruments)

        le = LabelEncoder()
        le.fit(self.data["instrument"].unique())
        self.data["instrument"] = le.transform(self.data["instrument"])
        self.ins_encoder = le
        print("All instruments:", le.classes_)
    """

    def get_data_chunks(self):
        """Divide CSV file into chunks based on start_time and end_time
        Each CSV chunk corresponds to an audio segment of length chunk_len"""

        data_chunks = [
            self.data[
                (self.data["start_time"] >= i)
                & (self.data["start_time"] < i + self.chunk_len)
            ]
            for i in range(0, self.audio_len, self.chunk_len)
        ][:-1]  # Drop last chunk
        # print(len(data_chunks))

        return data_chunks

    def get_chunk_ins(self):
        """Return a list of prominent instruments from each segment"""

        data_chunks = self.get_data_chunks()
        ins_per_chunk = []
        for i, chunk in enumerate(data_chunks):
            dcopy = chunk.copy()
            dcopy["duration"] = dcopy.apply(
                lambda row: min(row["end_time"], (i + 1) * self.chunk_len - 1)
                - row["start_time"],
                axis=1,
            )

            # Get all instruments prominent in current audio segment
            chunk_ins_list = []
            for ins in dcopy["instrument"].unique():
                sum_ins_time = dcopy[dcopy["instrument"] == ins]["duration"].sum()
                # print(f"Instrument: {ins}\t\tDuration: {sum_ins_time / sr}s")
                if sum_ins_time >= self.chunk_len * self.ins_dur_threshold:
                    chunk_ins_list.append(ins)

            ins_per_chunk.append(chunk_ins_list)

        return ins_per_chunk


"""
sr = 44100
chunk_secs = 3
audio_secs = 251

data_utils = CsvContainer(
    "assets/examples/music_net/1728.csv", sr * chunk_secs, sr * audio_secs
)
chunk_len = data_utils.chunk_len
# data_chunks = data_utils.get_data_chunks()
# print(data_chunks[0].describe())
# print("-----------------")
# print(data_chunks[0]["instrument"].unique())

ins_per_chunk = np.array(data_utils.get_chunk_ins())
print("Shape:", ins_per_chunk.shape)
print("-----------------")
print(ins_per_chunk[0])

# print(to_categorical())
# """
