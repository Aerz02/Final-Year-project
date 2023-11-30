# Aerhan Srirangan's Final Project

# Imports

from pathlib import Path
import datetime as dt

import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import *
from keras.models import Sequential
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Specify tensorflow version
print("Tensorflow Version:", tf.__version__)

# Specify current directory
print("Current directory: ", Path.cwd())
# Specify the directory containing the dataset
DATASET_DIR = "UCF-Crime Dataset"
# Specify path to dataset directory
dataset_path = Path.cwd() / DATASET_DIR
# Get each category of the dataset
CLASSES = [d.name for d in dataset_path.iterdir() if d.is_dir()]
print(CLASSES)
# Specify the height and width to which each video frame will be resized in our dataset.
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64


# Fetch Dataset

# Video class
class Video:
    def __init__(self, name: str, frames, filepath: str, label: str, frame_count: int, duration: str):
        self.name = name
        self.frames = frames
        self.filepath = filepath
        self.label = label
        self.frame_count = frame_count
        self.duration = duration


# helper function to display the length of the video
def get_video_duration(frame_count: float, fps: float) -> str:
    """"
    Calculates duration of the video in hh:mm:ss format
    :param frame_count: number of frames in the video
    :param fps: frame rate of the video
    :return: a string representing the duration of the video in hh:mm:ss format
    """
    # Calculate the length of the video in seconds
    length = frame_count / fps

    # Calculate the duration in hours, minutes, and seconds
    hours, remainder = divmod(length, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Return the duration as a formatted string
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

# As the dataset is 37 GB, I will only use a tiny portion of it(10 videos per label).
# Otherwise, it will be too extensive to compute.


# extracting frames from video
def extract_frames(video) -> list:  # Try to find the correct type for video
    """
    Extracts frames from a video then normalises the frames
    Args:
        video: the video capture from openCV
    Returns:
        The list of normalised frames
    """
    # A list to store the frames of the video
    frames = []
    while video.isOpened():
        ret, frame = video.read()
        # If the frame was not successfully read, break out of the loop
        if not ret:
            break
        # Convert the frame from BGR into RGB format.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize the Frame to fixed height and width.
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        # Normalize the resized frame by dividing it with 255 so each pixel value then lies between 0 and 1
        normalised_frame = resized_frame / 255
        frames.append(normalised_frame)

    video.release()
    return frames


# trying to make the video length the same, I will split the videos into equal length
def split_video(video, interval) -> list:
    """
    Splits a video into separate parts given the interval of frames.
    Args:
        video: The video as an array.
        interval: The interval of frames.
    Returns:
        A list of the split videos.
    """
    split_videos = []
    start = 0
    count = 1
    while start < len(video):
        end = min(start + interval, len(video))
        print(f"start = {start}, end = {end} loop iteration {count}")
        part = np.array(video[start:end])
        split_videos.append(part)
        start = end + 1
        count += 1
    return split_videos


FILE_LIMIT = 20


# function to fetch dataset stored on hard drive as it's 37GB
# and checks if the videos in the dataset are usable or not
def fetch_dataset() -> list:
    videos = []
    for label in CLASSES:
        # Display the name of the class whose data is being extracted.
        print(f'Extracting Data of: {label}')
        label_path = dataset_path / label
        print("Label path:", label_path)
        files = [f for f in label_path.iterdir() if f.suffix == '.mp4']
        # Select the first 10 videos of each label
        file_count = 0
        for file in files:
            filepath = label_path / file
            # Check if the video file can be opened
            video = cv2.VideoCapture(str(filepath))
            if file_count == FILE_LIMIT:
                break
            if not video.isOpened():
                print(f"Could not open video file: {str(filepath)}")
                files.remove(file)
                continue
            else:
                fps = video.get(cv2.CAP_PROP_FPS)
                frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
                duration = get_video_duration(frame_count, fps)
                print(file, "is ", duration)
                # Extract the frames of the video
                frames = extract_frames(video)
                # the smallest video in the dataset is 139 frames long
                if int(frame_count) % 139 == 0:
                    # Add video to video array
                    video = Video(file, frames, filepath, label, int(frame_count), duration)
                    videos.append(video)
                    file_count += 1
    return videos


# Video data processing

videos = fetch_dataset()

frame_counts = [video.frame_count for video in videos]

print("Number of videos", len(videos))

min_frames = min(frame_counts)
max_frames = max(frame_counts)

print("max number of frames in all the videos", max_frames)
print("min number of frames in all the videos", min_frames)

video_labels = np.array([video.label for video in videos])
print("Shape of video labels:", video_labels.shape)
print("Data type of video labels", video_labels.dtype)
features = np.asarray([video.frames for video in videos])
print("Shape of features:", features.shape)
print("Data type of features", features.dtype)

print(f"features has length of {len(features)}")
print(f"Video Labels has length of {len(video_labels)}")

# elif int(frame_count) > 139:
#     video_segments = split_video(frames, 139)
#     for i in range(len(video_segments)):
#     video = Video(str(str(file) + "Part "+ str(i + 1)), video_segments[i],filepath, label,
#                   len(video_segments[i]), get_video_duration(len(video_segments[i]), fps))
#                             videos.append(video)
# durations = [video.duration for video in videos]
# print(durations)

# Create Test and Train Set

# splitting into train and test sets
features_train, features_test, labels_train, labels_test = train_test_split(features,
                                                                            video_labels,
                                                                            test_size=0.2, shuffle=True)

features_train = np.array(features_train)
features_test = np.array(features_test)

# Encoding labels

# Convert labels to numerical categories
# Reshape the array to 2D
labels_train = labels_train.reshape(-1, 1)
labels_train = np.array(labels_train)

labels_test = labels_test.reshape(-1, 1)
labels_test = np.array(labels_test)

# Create an instance of the OneHotEncoder
encoder = OneHotEncoder()

# # Fit the encoder to the data and transform the data
# encoded_labels_train = encoder.fit_transform(labels_train)
# encoded_labels_train = np.array(encoded_labels_train.reshape(-1, 1))
# # encoded_labels_train = tf.convert_to_tensor(encoded_labels_train.reshape(-1, 1))

# encoded_labels_test = encoder.fit_transform(labels_test)
# encoded_labels_test = np.array(encoded_labels_train).reshape(-1, 1)
# encoded_labels_test = np.array(encoded_labels_test)
encoded_labels_train = encoder.fit_transform(labels_train).toarray()
encoded_labels_test = encoder.fit_transform(labels_test).toarray()

#I'll be using One Hot Encoding as it is more flexible and since there is no natural order or hierarchy

# Create ConvLSTM model

def create_conv_lstm_model():
    model = Sequential()

    model.add(ConvLSTM2D(filters=4, kernel_size=(3, 3), activation='tanh', data_format="channels_last",
                         return_sequences=True, input_shape=(min_frames, IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    model.add(TimeDistributed(Dropout(0.05)))

    model.add(ConvLSTM2D(filters=8, kernel_size=(3, 3), activation='tanh', data_format="channels_last",
                         return_sequences=True))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    model.add(TimeDistributed(Dropout(0.05)))

    model.add(ConvLSTM2D(filters=14, kernel_size=(3, 3), activation='tanh', data_format="channels_last",
                         return_sequences=True))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    model.add(TimeDistributed(Dropout(0.05)))

    model.add(ConvLSTM2D(filters=16, kernel_size=(3, 3), activation='tanh', data_format="channels_last",
                         return_sequences=True))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))

    model.add(Flatten())

    model.add(Dense(len(CLASSES), activation="softmax"))

    # Display the models summary.
    model.summary()

    # Return the constructed convlstm model.
    return model


# Construct the required convlstm model.
conv_lstm_model = create_conv_lstm_model()
# Display the success message.
print("Model Created Successfully!")
# Plot the structure of the constructed model.
plot_model(conv_lstm_model, to_file='conv_lstm_model_structure_plot.png', show_shapes=True, show_layer_names=True)

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)

# Compile the model and specify loss function, optimizer and metrics values to the model
conv_lstm_model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=["accuracy"])

# Training Model

print("Shape of features_train:", np.shape(features_train))
print("Type of features_train:", type(features_train))
print("Type of encoded training labels:", type(encoded_labels_train))
print("Shape of encoded training labels:", np.shape(encoded_labels_train))

# Start training the model.
conv_lstm_model_training_history = conv_lstm_model.fit(features_train, encoded_labels_train,
                                                       epochs=50, batch_size=4, shuffle=True,
                                                       callbacks=[early_stopping_callback])

# Evaluate model

# Evaluate the trained model.
model_evaluation_history = conv_lstm_model.evaluate(features_test, labels_test)

# Save Model

# Get the loss and accuracy from model_evaluation_history.
model_evaluation_loss, model_evaluation_accuracy = model_evaluation_history

# Define the string date format.
# Get the current Date and Time in a DateTime Object.
# Convert the DateTime object to string according to the style mentioned in date_time_format string.
date_time_format = '%Y_%m_%d__%H_%M_%S'
current_date_time_dt = dt.datetime.now()
current_date_time_string = dt.datetime.strftime(current_date_time_dt, date_time_format)

# Define a useful name for our model to make it easy for us while navigating through multiple saved models.
model_file_name = f'conv_lstm_model___Date_Time_' \
                  f'{current_date_time_string}___Loss_' \
                  f'{model_evaluation_loss}___Accuracy_{model_evaluation_accuracy}.png'

# Save your Model.
conv_lstm_model.save(model_file_name)


# Plots

def plot_metric(model_training_history, metric_name_1, metric_name_2, plot_name):
    """
    This function will plot the metrics passed to it in a graph.
    Args:
        model_training_history: A history object containing a record of training and validation
                                loss values and metrics values at successive epochs
        metric_name_1:          The name of the first metric that needs to be plotted in the graph.
        metric_name_2:          The name of the second metric that needs to be plotted in the graph.
        plot_name:              The title of the graph.
    """

    # Get metric values using metric names as identifiers.
    metric_value_1 = model_training_history.history[metric_name_1]
    metric_value_2 = model_training_history.history[metric_name_2]

    # Construct a range object which will be used as x-axis (horizontal plane) of the graph.
    epochs = range(len(metric_value_1))

    # Plot the Graph.
    plt.plot(epochs, metric_value_1, 'blue', label=metric_name_1)
    plt.plot(epochs, metric_value_2, 'red', label=metric_name_2)

    # Add title to the plot.
    plt.title(str(plot_name))

    # Add legend to the plot.
    plt.legend()


# Visualize the training and validation loss metrics.
plot_metric(conv_lstm_model_training_history, 'loss', 'val_loss', 'Total Loss vs Total Validation Loss')

# Visualize the training and validation accuracy metrics.
plot_metric(conv_lstm_model_training_history, 'accuracy', 'val_accuracy',
            'Total Accuracy vs Total Validation Accuracy')
