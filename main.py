import os

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
import absl.logging

# turn off warnings
absl.logging.set_verbosity(absl.logging.ERROR)

max_length = 230  # 308
n_mfcc = 40
num_classes = 7


# 1311  -     / 426
def generate_padded_spectrogram(input_file, output_file, max_length, save_image=False):
    # Load the audio file
    y, sr = librosa.load(input_file)
    y, _ = librosa.effects.trim(y)

    # Compute the spectrogram
    # D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
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
            if num_classes == 6 and emotion == "Neutral":
                continue

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

    np.savez(f'spectrograms/spectrogram_data{num_classes}.npz', labels=labels_one_hot, features=np.array(features))
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
    loaded_data = np.load(f'spectrograms/spectrogram_data{num_classes}.npz')

    # Access the arrays using the keys provided during saving
    features = loaded_data['features']
    labels = loaded_data['labels']

    return features, labels


def train(config=None):
    with wandb.init(config=config):
        if not os.path.exists(f"spectrograms/spectrogram_data{num_classes}.npz"):
            X_data, y_data = generate_all_spectrograms()
        else:
            X_data, y_data = load_spectrogram_data()

        config = wandb.config
        x_train, x_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.1, random_state=42)
        model = Sequential()

        model.add(LSTM(config.layer_size_1, input_shape=X_data.shape[1:], return_sequences=True))
        model.add(LSTM(config.layer_size_2))
        model.add(Dense(config.layer_size_3, activation=config.activation_3))
        model.add(Dropout(config.dropout))
        model.add(Dense(num_classes, activation=config.activation_4))

        # Compile the model
        model.compile(loss=config.loss, optimizer=config.optimizer, metrics=[config.metric])

        # Train the model
        history = model.fit(x_train, y_train, epochs=config.epoch, batch_size=config.batch_size, validation_split=0.2,
                            callbacks=[WandbMetricsLogger(log_freq=5),
                                       WandbModelCheckpoint("models")])

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

        model.save(f'model_saves/{wandb.run.name}.h5')
        # Finish the wandb run
        wandb.finish()

def new_ml():
    config = {
        "layer_size_1": 128,
        "layer_size_2": 64,
        "layer_size_3": 128,
        "activation_3": "relu",
        "dropout": 0.4,
        "activation_4": "softmax",
        "optimizer": "adam",
        "loss": "categorical_crossentropy",
        "metric": ["accuracy"],
        "epoch": 10,
        "batch_size": 32
    }
    train(config)


if __name__ == "__main__":
    # new_ml()

    if not os.path.exists(f"spectrograms/spectrogram_data{num_classes}.npz"):
        X_data, y_data = generate_all_spectrograms()
    else:
        X_data, y_data = load_spectrogram_data()

    run = wandb.init()

    model = keras.models.load_model('artifacts/run_o2b824li_model-v9' if num_classes == 6 else 'model_saves/feasible-smoke-11.h5')

    modified_model = tf.keras.Sequential(model.layers[:-1])

    # Add a new layer
    modified_model.add(Dense(num_classes, activation='softmax', name='new_output_layer'))  # Example new layer

    # Freeze existing layers
    for layer in modified_model.layers[:-1]:
        layer.trainable = False

    x_train, x_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.1, random_state=42)

    # Compile the model
    modified_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Train the model
    history = modified_model.fit(x_train, y_train, epochs=3, batch_size=32, validation_split=0.2,
                        callbacks=[WandbMetricsLogger(log_freq=1),
                                   WandbModelCheckpoint("models")])

    accuracy = modified_model.evaluate(x_test, y_test)[1]
    print(f'Experiment for classification for model trained on {num_classes} emotions used to classify {num_classes + 1 if num_classes == 6 else num_classes - 1} emotions')
    print(f"Test Accuracy: {accuracy * 100}%")

    # Evaluate the model on the test set
    accuracy_train = modified_model.evaluate(x_train[0:1000], y_train[0:1000])[1]
    print(f"Test Accuracy: {accuracy_train * 100}%")

    # Log test results using wandb
    wandb.log({'test_accuracy': accuracy * 100, 'subset_train_accuracy': accuracy_train * 100})

    wandb.finish()
