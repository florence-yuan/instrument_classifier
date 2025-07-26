import audio_utils
import csv_utils
from config_utils import instruments_map

import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import itertools

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import pickle


class Model:
    def __init__(self):
        self.chunk_secs = 3
        self.min_confidence = 0.6  # Minimum instrument confidence

    def init_model(self, filenames):
        self.tot_audio_len = 0
        # self.sr = 44100
        inputs, outputs = [], []
        for i, fn in enumerate(filenames):
            audio = "assets/musicnet/audio/" + fn + "wav"
            y, sr = librosa.load(audio, sr=None)
            self.tot_audio_len += len(y) / sr

            audio_container = audio_utils.AudioContainer(file_path=audio)

            csv_container = csv_utils.CsvContainer(
                file_path="assets/musicnet/csv/" + fn + "csv",
                chunk_len=self.chunk_secs * sr,
                audio_len=len(y),
            )

            spec_chunks = audio_container.get_spec_chunks()
            ins_per_chunk = csv_container.get_chunk_ins()
            inputs.append(spec_chunks)
            outputs.append(ins_per_chunk)

            # print(len(outputs), len(spec_chunks), len(y))

        outputs = list(itertools.chain.from_iterable(outputs))

        # Binarize Instruments
        mlb = MultiLabelBinarizer()
        outputs = mlb.fit_transform(outputs)
        self.ins_encoder = mlb

        # print("\n", outputs[:10])

        with open("models/ins_binarizer.pkl", "wb") as file:
            pickle.dump(mlb, file)

        self.X = np.array(np.vstack(inputs))
        self.y = np.array(outputs)

    def ins_idx_to_name(self, idx_list):
        """Utility function for converting instrument indexes to names"""

        return list(map(lambda idx: instruments_map[int(idx)], idx_list))

    def print_attrs(self):
        print("\n\n")
        print("Total audio length\t", self.tot_audio_len)
        print(
            "Instruments\t\t",
            self.ins_idx_to_name(self.ins_encoder.classes_),
        )
        print("\n\n")

    def get_shapes(self):
        print("\n\n")
        print("X shape:", self.X.shape)
        print("y shape:", self.y.shape)
        print("\n\n")

    def train_model(self, save_model=True):
        """Initialize, compile, and train model"""

        output_sz = self.y.shape[1]
        model = keras.Sequential(
            [
                Input(shape=self.X.shape[1:]),
                Conv2D(
                    filters=8,
                    kernel_size=4,
                    # input_shape=self.X.shape[1:],
                    activation="relu",
                ),
                MaxPooling2D(pool_size=2),
                Flatten(),
                Dense(output_sz, activation="sigmoid"),
            ]
        )
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["binary_accuracy"]
        )

        np.random.seed(10)
        tf.random.set_seed(10)

        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(
            self.X, self.y
        )

        self.history = model.fit(
            self.train_X,
            self.train_y,
            epochs=50,
            validation_data=(self.test_X, self.test_y),
        )

        self.model = model

        if save_model:
            model.save("models/model_small.keras")

    def load_model(self):
        """Load saved model"""

        model = keras.models.load_model("models/model_small.keras")

        with open("models/ins_binarizer.pkl", "rb") as file:
            self.ins_encoder = pickle.load(file)

        self.model = model

    def vis_training(self):
        """Visualize model accuracy"""

        pd.DataFrame(self.history.history)[
            ["binary_accuracy", "val_binary_accuracy"]
        ].plot(figsize=(10, 5))
        plt.show()

    def process_preds(self, preds):
        """Convert predictions from sigmoid output to 0s and 1s"""

        return np.where(preds >= self.min_confidence, 1, 0)

    def predict(self, audio):
        """Predict instruments present in new audio"""

        audio_container = audio_utils.AudioContainer(file_path=audio)
        spec_chunks = np.array(audio_container.get_spec_chunks())

        preds = self.model.predict(spec_chunks)
        # print("\n\nPredictions:")
        # print(preds[:10])
        preds = self.process_preds(preds)

        has_ins = np.any(preds, axis=0)
        num_ins = np.count_nonzero(has_ins)
        print("Number of instruments:", num_ins)
        print(
            "Unique instruments:\t",
            self.ins_idx_to_name(
                self.ins_encoder.inverse_transform(np.array([has_ins]))[0]
            ),
        )

    def predict_from_test(self):
        num_tests = 5
        preds = self.model.predict(self.test_X[:num_tests])
        print("Predictions:\t", preds)
        print("Actual:\t\t", self.test_y[:num_tests])


def build_train_model():
    """Fetches files for training, builds and trains model"""

    filenames = []
    audio_dir_path = "assets/musicnet/audio/"

    for file_path in os.listdir(audio_dir_path):
        filenames.append(file_path.rstrip("wav"))

    model = Model()
    model.init_model(filenames=filenames)
    model.train_model()
    model.vis_training()
    #
    # model.get_shapes()    # X_shape='(505, 128, 258, 1)'
    #                       # y_shape='(505, 5)'


# build_train_model()
