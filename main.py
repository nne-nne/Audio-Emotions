import os

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
max_length = 315700


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


def generate_padded_spectrogram(input_file, output_file, max_length):
    # Load the audio file
    y, sr = librosa.load(input_file)

    # Compute the spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    # Pad or truncate the spectrogram to the desired length
    padded_D = pad_sequences([D.T], maxlen=max_length, padding='post', truncating='post', dtype='float32')[0].T

    # Plot the spectrogram without labels and margins
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(padded_D, y_axis='log', x_axis=None)
    plt.axis('off')  # Turn off axis labels
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
    plt.close()


def get_size(_input_file):
    # Load the audio file
    y, sr = librosa.load(_input_file)

    # Generate a spectrogram
    d = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    return d.size


def generate_all_spectrograms():
    input_root = 'res/AudioEmotions/Emotions'
    output_root = 'spectrograms'

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
                    generate_padded_spectrogram(audio_file_path, output_file_path, max_length)


def get_max_size():
    input_root = 'res/AudioEmotions/Emotions'
    max_size = 0

    # Iterate over all emotion folders
    for emotion in os.listdir(input_root):
        emotion_path = os.path.join(input_root, emotion)

        # Check if it's a directory
        if os.path.isdir(emotion_path):

            # Iterate over audio files in the emotion folder
            for audio_file in os.listdir(emotion_path):
                if audio_file.endswith('.wav'):  # You can modify this condition based on your audio file format
                    audio_file_path = os.path.join(emotion_path, audio_file)
                    size = get_size(audio_file_path)
                    if size > max_size:
                        max_size = size
    return max_size


if __name__ == "__main__":
    generate_all_spectrograms()
# generate_spectrogram_chart("./res/AudioEmotions/Emotions/Sad/OAF_king_sad.wav", "spectrograms/d.png")
