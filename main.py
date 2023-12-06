import os
from datetime import datetime

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

max_length = 230  # 308
n_mfcc = 40
num_classes = 7


def generate_spectrogram_chart(_input_file, _output_file):
    # Load the audio file
    y, sr = librosa.load(_input_file)

    # Generate a spectrogram
    d = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    # Plot the spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(d, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.savefig(_output_file, bbox_inches='tight')
    plt.close()


def generate_spectrogram(_input_file, _output_file):
    # Load the audio file
    y, sr = librosa.load(_input_file)

    # Generate a spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    # Plot the spectrogram without labels and margins
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, y_axis='log', x_axis=None)
    plt.axis('off')  # Turn off axis labels
    plt.savefig(_output_file, bbox_inches='tight', pad_inches=0)
    plt.close()


# 1311  -     / 426
def generate_padded_spectrogram(input_file, output_file, max_length, save_image=False):
    # Load the audio file
    y, sr = librosa.load(input_file)
    y, _ = librosa.effects.trim(y)

    # Compute the spectrogram
    #D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    D = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # Pad or truncate the spectrogram to the desired length
    padded_D = keras.utils.pad_sequences([D.T], maxlen=max_length, padding='post', truncating='post', dtype='float32', value=0)[0].T

    # Plot the spectrogram without labels and margins
    if save_image:
        plt.ioff()
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(padded_D, y_axis='log', x_axis=None)
        plt.axis('off')  # Turn off axis labels
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
        plt.close()

    return padded_D


def get_size(_input_file):
    # Load the audio file
    y, sr = librosa.load(_input_file)

    # Generate a spectrogram
    d = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    return d.size


def generate_all_spectrograms():
    input_root = 'res/AudioEmotions/Emotions'
    output_root = 'spectrograms'

    features = []
    labels = []

    # Iterate over all emotion folders
    for emotion in os.listdir(input_root):
        emotion_path = os.path.join(input_root, emotion)

        # Check if it's a directory
        if os.path.isdir(emotion_path):

            # Create output directory for the current emotion
            output_dir = os.path.join(output_root, emotion)
            os.makedirs(output_dir, exist_ok=True)

            # Iterate over audio files in the emotion folder
            for audio_file in os.listdir(emotion_path):
                if audio_file.endswith('.wav'):  # You can modify this condition based on your audio file format
                    audio_file_path = os.path.join(emotion_path, audio_file)
                    output_file_path = os.path.join(output_dir, os.path.splitext(audio_file)[0] + '.png')
                    feature = generate_padded_spectrogram(audio_file_path, output_file_path, max_length)
                    features.append(feature)
                    labels.append(emotion)

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    labels_one_hot = keras.utils.to_categorical(encoded_labels, num_classes)

    np.savez('spectrograms/spectrogram_data.npz', labels=labels_one_hot, features=np.array(features))
    return np.array(features), labels_one_hot


def get_max_size():
    input_root = 'res/AudioEmotions/Emotions'
    max_size = (0, 0)
    lengths = []

    # Iterate over all emotion folders
    for emotion in os.listdir(input_root):
        emotion_path = os.path.join(input_root, emotion)

        # Check if it's a directory
        if os.path.isdir(emotion_path):

            # Iterate over audio files in the emotion folder
            for audio_file in os.listdir(emotion_path):
                if audio_file.endswith('.wav'):  # You can modify this condition based on your audio file format
                    audio_file_path = os.path.join(emotion_path, audio_file)
                    y, sr = librosa.load(audio_file_path)
                    y, _ = librosa.effects.trim(y)
                    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
                    size = D.shape

                    lengths.append(size[1])
                    if size[1] > max_size[1]:
                        max_size = (size[0], size[1])

    print("Total Spectrograms:", len(lengths))
    print("Min length:", min(lengths))
    print("Max length:", max(lengths))
    print("Mean length:", np.mean(lengths))
    print("Median length:", np.median(lengths))

    plt.figure(figsize=(10, 5))
    plt.hist(lengths, bins=20, color='blue', edgecolor='black', alpha=0.7)
    plt.title('Histogram of Spectrogram lengths')
    plt.xlabel('length')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    return max_size

def load_spectrogram_data():
    loaded_data = np.load('spectrograms/spectrogram_data.npz')

    # Access the arrays using the keys provided during saving
    features = loaded_data['features']
    labels = loaded_data['labels']

    return features, labels

if __name__ == "__main__":
    if not os.path.exists("spectrograms/spectrogram_data.npz"):
        X_data, y_data = generate_all_spectrograms()
    else:
        X_data, y_data = load_spectrogram_data()

    # Initialize wandb
    wandb.init(
        # set the wandb project where this run will be logged
        project="audioemo",

        # track hyperparameters and run metadata with wandb.config
        config={
            "layer_1": 128,
            "activation_1": "tanh",
            "layer_2": 64,
            "activation_2": "tanh",
            "layer_3": 256,
            "activation_3": "relu",
            "dropout": 0.5,
            "layer_5": 256,
            "activation_5": "softmax",
            "optimizer": "adam",
            "loss": "categorical_crossentropy",
            "metric": "accuracy",
            "epoch": 20,
            "batch_size": 32
        }
    )

    x_train, x_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.1, random_state=42)
    model = Sequential()

    model.add(LSTM(128, input_shape=X_data.shape[1:], return_sequences=True))
    model.add(LSTM(64))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    history = model.fit(x_train, y_train, epochs=20, batch_size=32, validation_split=0.2, callbacks=[WandbMetricsLogger(log_freq=5),
                      WandbModelCheckpoint("models")])

    # Log hyperparameters using wandb
    # wandb.config.epochs = 20
    # wandb.config.batch_size = 32

    # Log training results using wandb
    wandb.log({'train_loss': history.history['loss'][-1], 'train_accuracy': history.history['accuracy'][-1]})
    wandb.log({'val_loss': history.history['val_loss'][-1], 'val_accuracy': history.history['val_accuracy'][-1]})

    # Evaluate the model on the test set
    accuracy = model.evaluate(x_test, y_test)[1]
    print(f"Test Accuracy: {accuracy * 100}%")

    # Evaluate the model on the test set
    accuracy_train = model.evaluate(x_train[0:1000], y_train[0:1000])[1]
    print(f"Test Accuracy: {accuracy_train * 100}%")

    # Log test results using wandb
    wandb.log({'test_accuracy': accuracy * 100, 'subset_train_accuracy': accuracy_train * 100})

    # Finish the wandb run
    wandb.finish()
