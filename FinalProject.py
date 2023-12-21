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
dataset_path = Path.cwd().parent.parent / "Documents/Uni/Year 3/Final Year Project/Model" / DATASET_DIR
# Get each category of the dataset
labels = [d.name for d in dataset_path.iterdir() if d.is_dir()]
print(labels)
# Specify the height and width to which each video frame will be resized in our dataset.
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64


# Dataset Analysis

# Video Length function
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


# Get Factors function
def get_factors(n):
    factors = []
    for i in range(2, n + 1):
        if n % i == 0:
            factors.append(i)
    return factors


# Given the minutes and fps, find the amount of frame
def minutes_to_frames(minutes: int, fps: int) -> int:
    seconds = minutes * 60
    return seconds * fps


def get_dataset_info():
    """
    Function to iterate through videos, calculate factors of frame count,
    and create a Pandas DataFrame with filepath, label, frame count, frames and factors of frame count.

    Arguments:
    dataset_path: str - Path to the dataset directory.

    Returns:
    pandas.DataFrame - DataFrame containing filepath, label, frame count, and factors of frame count.
    """
    videos = []
    for label in labels:
        label_path = dataset_path / label
        files = [f.name for f in label_path.iterdir() if f.suffix == '.mp4']
        for file in files:
            filepath = label_path / file
            video = cv2.VideoCapture(str(filepath))
            if not video.isOpened():
                video.release()
                print(f"Could not open video file: {str(filepath)}")
                continue
            fps = int(video.get(cv2.CAP_PROP_FPS))
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            # Calculate factors of the frame count
            factors = get_factors(frame_count)
            video_info = {
                'File': str(file),
                'Label': label,
                'Frame Count': frame_count,
                'FPS': fps,
                "Duration": get_video_duration(frame_count, fps),
                'Factors of Frame Count': factors
            }
            print(str(file))
            videos.append(video_info)
    # Create a DataFrame from the collected information
    df = pd.DataFrame(videos)
    # Converts the dataframe into str object with formatting
    print(df)
    return df


videos_info = get_dataset_info()


def factor_count_per_label(df, label):
    """
    Function to count the number of videos per factor for a specific label.

    Arguments:
    df: pandas.DataFrame - DataFrame containing filepath, label, frame count, and factors.
    label: str - The label for which factor count is calculated.

    Returns:
    pd.DataFrame - DataFrame showing factor counts for the specified label, ordered from most shared to least shared.
    """
    label_df = df[df['Label'] == label]
    factor_counts = {}

    for factors_list in label_df['Factors of Frame Count']:
        for factor in factors_list:
            factor_counts[factor] = factor_counts.get(factor, 0) + 1

    factor_counts_df = (pd.DataFrame(factor_counts.items(),
                                     columns=['Factor of Frame Count', 'Number of Videos That Share The Factor'])
                        .sort_values(by='Number of Videos That Share The Factor', ascending=False))

    return factor_counts_df


# Iterate over each label to create a DataFrame for each label showing factor counts
label_factor_counts = {}
for label in labels:
    print(label)
    label_factor_counts[label] = factor_count_per_label(videos_info, label)
    print(label_factor_counts[label])

frame_counts = videos_info["Frame Count"]

print("Number of videos", len(videos_info))

min_frames = min(frame_counts)
max_frames = max(frame_counts)
print("max number of frames in all the videos", max_frames)
print("Duration of max_frames is " + get_video_duration(max_frames, 30))
print("min number of frames in all the videos", min_frames)
print("Duration of min_frames is " + get_video_duration(min_frames, 30))


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

    def __str__(self):
        return f"{self.name} has {self.frame_count} frames and is {self.duration}"


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
        resized_frame = cv2.resize(rgb_frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
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
    length = len(video)
    for i in range(0, length, interval):
        # print(f"i = {i}, interval = {interval}, i + interval = {i + interval}, video length = {len(video)}")
        segment = video[i:i + interval]
        if len(segment) == interval:
            split_videos.append(segment)
        else:
            print(f"Incorrect size: Segment length = {len(segment)}, start = {i}, end = {i + interval}"
                  f", video length = {len(video)}")
    return split_videos


# function to fetch dataset stored on hard drive as it's 37GB
# and checks if the videos in the dataset are usable or not
def fetch_ucf_crime_dataset():
    FRAMES_LIMIT = minutes_to_frames(3, 30)
    FILE_LIMIT = 10
    videos = []
    for label in labels:
        label_path = dataset_path / label
        files = [f.name for f in label_path.iterdir() if f.suffix == '.mp4']
        file_count = 0
        for file in files:
            if file_count == FILE_LIMIT:  # Break outer loop if the file count limit is reached
                break
            filepath = label_path / file
            video_cap = cv2.VideoCapture(str(filepath))
            if not video_cap.isOpened():
                video_cap.release()
                print(f"Could not open video file: {str(file)}")
                continue
            frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count >= FRAMES_LIMIT:
                video_cap.release()
                continue
            fps = int(video_cap.get(cv2.CAP_PROP_FPS))
            frames = extract_frames(video_cap)
            duration = get_video_duration(frame_count, fps)
            if len(frames) == frame_count:
                if len(frames) % 2 == 0:
                    print(f"{str(file)} has {frame_count} frames at {fps} fps and is {duration} long")
                    file_count += 1
                    if frame_count > 2:
                        video_segments = split_video(frames, 2)
                        print(f"Video number {file_count}")
                        for i in range(len(video_segments)):
                            new_video = Video(str(str(file) + " Part " + str(i + 1)), video_segments[i],
                                              str(filepath), label, len(video_segments[i]),
                                              get_video_duration(len(video_segments[i]), fps))
                            videos.append(new_video)
            else:
                # print(f"Error: lengths are not equal, {file}")
                continue
    return videos


# Video data processing

video_data = fetch_ucf_crime_dataset()

# for i in video_data:
#     print(str(i))
frame_counts = [video.frame_count for video in video_data]

print("Number of videos", len(video_data))

# Size and Duration of the all videos stored
print(f"Number of frames of all videos together is {sum(frame_counts)}")
print(f"Duration of all videos together is {get_video_duration(sum(frame_counts), 30)}")

video_labels = np.array([video.label for video in video_data])
features = np.asarray([video.frames for video in video_data])
print("Shape of features:", features.shape)
print("Data type of features", features.dtype)

print(f"features has length of {len(features)}")
print(f"Video Labels has length of {len(video_labels)}")

min_frames = min(frame_counts)
max_frames = max(frame_counts)
print(f"the smallest amount of frames is {min_frames}")
print(f"the largest amount of frames is {max_frames}")

# Encoding labels

# I'll be using One Hot Encoding as it is more flexible and since there is no natural order or hierarchy
# Convert labels to numerical categories

# Initialize OneHotEncoder
one_hot_encoder = OneHotEncoder()

# Reshape the video_labels array to 2D shape
video_labels_reshaped = video_labels.reshape(-1, 1)

# Fit and transform the labels to one-hot encoded vectors
one_hot_encoded_labels = one_hot_encoder.fit_transform(video_labels_reshaped).toarray()


# Create Test and Train Set
# splitting into train and test sets
features_train, features_test, labels_train, labels_test = train_test_split(features,
                                                                            one_hot_encoded_labels,
                                                                            test_size=0.2, shuffle=True)

features_train = np.array(features_train)
features_test = np.array(features_test)


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

    model.add(Dense(len(labels), activation="softmax"))

    # Display the models summary.
    model.summary()

    # Return the constructed ConvLSTM model.
    return model


# Construct the required ConvLSTM model.
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
print("Type of encoded training labels:", type(labels_train))
print("Shape of encoded training labels:", np.shape(labels_train))


# Start training the model.
conv_lstm_model_training_history = conv_lstm_model.fit(x=features_train, y=labels_train, epochs=50,
                                                       batch_size=4, shuffle=True, validation_split=0.2,
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
date_time_format = '%d/%m/%Y @ %H:%M:%S'
current_date_time_dt = dt.datetime.now()
current_date_time_string = dt.datetime.strftime(current_date_time_dt, date_time_format)

# Define a useful name for our model to make it easy for us while navigating through multiple saved models.
model_file_name = f'ConvLSTM_model_{current_date_time_string}'

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


# save plots to drive
def save_plot(filename: str, extension="png", directory="./"):
    # Create a Path object for the directory
    directory_path = Path(directory)

    # Create the directory if it doesn't exist
    directory_path.mkdir(parents=True, exist_ok=True)

    # Create a Path object for the file
    file_path = directory_path / f"{filename}.{extension}"

    # Save the plot using the Path object
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {file_path}")


# Visualize the training and validation loss metrics.
plot_metric(conv_lstm_model_training_history, 'loss', 'val_loss', 'Total Loss vs Total Validation Loss')

save_plot(str(model_file_name + " Total Loss vs Total Validation Loss"), "png", model_file_name)

# Visualize the training and validation accuracy metrics.
plot_metric(conv_lstm_model_training_history, 'accuracy', 'val_accuracy',
            'Total Accuracy vs Total Validation Accuracy')

save_plot(str(model_file_name + " Total Accuracy vs Total Validation Accuracy"), "png", model_file_name)
