import argparse
import os
import pathlib

import numpy as np
import keras
import keras.applications as kapp
from PIL import Image, ExifTags
import random
import scipy.spatial
import pandas as pd
from collections import namedtuple

# Standard PySceneDetect imports:
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.scene_manager import generate_images
# For caching detection metrics and saving/loading to a stats file
from scenedetect.stats_manager import StatsManager

# For content-aware scene detection:
from scenedetect.detectors.content_detector import ContentDetector


# Define a named tuple to keep our image information together
ImageFile = namedtuple("ImageFile", "src filename path uri")

def find_scenes(video_path):
    # type: (str) -> List[Tuple[FrameTimecode, FrameTimecode]]
    video_manager = VideoManager([video_path])
    stats_manager = StatsManager()
    # Construct our SceneManager and pass it our StatsManager.
    scene_manager = SceneManager(stats_manager)

    # Add ContentDetector algorithm (each detector's constructor
    # takes detector options, e.g. threshold).
    scene_manager.add_detector(ContentDetector())
    base_timecode = video_manager.get_base_timecode()

    # We save our stats file to {VIDEO_PATH}.stats.csv.
    stats_file_path = '%s.stats.csv' % video_path

    scene_list = []

    try:
        # If stats file exists, load it.
        if os.path.exists(stats_file_path):
            # Read stats from CSV file opened in read mode:
            with open(stats_file_path, 'r') as stats_file:
                stats_manager.load_from_csv(stats_file, base_timecode)

        # Set downscale factor to improve processing speed.
        video_manager.set_downscale_factor()

        # Start video_manager.
        video_manager.start()

        # Perform scene detection on video_manager.
        scene_manager.detect_scenes(frame_source=video_manager)

        # Obtain list of detected scenes.
        scene_list = scene_manager.get_scene_list(base_timecode)
        # Each scene is a tuple of (start, end) FrameTimecodes.

        video_file_name = video_path.split(".")[-2]
        video_file_process_path = "temp_video_frames/"+video_file_name
        generate_images(scene_list, video_manager, video_file_process_path, num_images=1)

    finally:
        video_manager.release()

    return scene_list


# Define a new filter for jinja to get the name of the file from a path
def basename(text):
    return text.split(os.path.sep)[-1]


def is_image(filename):
    """ Checks the extension of the file to judge if it an image or not. """
    fn = filename.lower()
    return fn.endswith("jpg") or fn.endswith("jpeg") or fn.endswith("png")


def find_image_files(root):
    """ Starting at the root, look in all the subdirectories for image files. """
    file_names = []
    for path, _, files in os.walk(os.path.expanduser(root)):
        for name in files:
            if is_image(name):
                file_names.append(os.path.join(path, name))
    return file_names


def build_model(model_name):
    """ Create a pretrained model without the final classification layer. """
    if model_name == "resnet50":
        model = kapp.resnet50.ResNet50(weights="imagenet", include_top=False)
        return model, kapp.resnet50.preprocess_input
    elif model_name == "vgg16":
        model = kapp.vgg16.VGG16(weights="imagenet", include_top=False)
        return model, kapp.vgg16.preprocess_input
    else:
        raise Exception("Unsupported model error")


def fix_orientation(image):
    """ Look in the EXIF headers to see if this image should be rotated. """
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == "Orientation":
                break
        exif = dict(image._getexif().items())

        if exif[orientation] == 3:
            image = image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image = image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image = image.rotate(90, expand=True)
        return image
    except (AttributeError, KeyError, IndexError):
        return image


def extract_center(image):
    """ Most of the models need a small square image. Extract it from the center of our image."""
    width, height = image.size
    new_width = new_height = min(width, height)

    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2

    return image.crop((left, top, right, bottom))


def process_images(file_names, preprocess_fn):
    """ Take a list of image filenames, load the images, rotate and extract the centers, 
    process the data and return an array with image data. """
    image_size = 224
    print_interval = len(file_names) / 10

    image_data = np.ndarray(shape=(len(file_names), image_size, image_size, 3))
    for i, ifn in enumerate(file_names):
        im = Image.open(ifn.src)
        im = fix_orientation(im)
        im = extract_center(im)
        im = im.resize((image_size, image_size))
        im = im.convert(mode="RGB")
        filename = os.path.join(ifn.path, ifn.filename + ".jpg")
        im.save(filename)
        image_data[i] = np.array(im)
        if i % print_interval == 0:
            print("Processing image:", i, "of", len(file_names))
    return preprocess_fn(image_data)


def generate_features(model, images):
    return model.predict(images)


def calculate_distances(features):
    return scipy.spatial.distance.cdist(features, features, "cosine")


def generate_site(output_path, names, features):
    """ Take the features and image information. Find the closest features and
    and generate static html files with similar images."""

    # Calculate all pairwise distances
    distances = calculate_distances(features)

    list_of_similar_images = []

    # Go through each image, sort the distances and generate the html file
    for idx, ifn in enumerate(names):
        # output_filename = os.path.join(output_path, ifn.filename + ".html")

        dist = distances[idx]
        images_confidence =  distances[idx]
        images_confidence = np.sort(images_confidence)[::-1]
        images_confidence = images_confidence[:15]
        similar_image_indexes = np.argsort(dist)[:15]
        similar_images = [names[i] for i in similar_image_indexes if i != idx]

        for row in range(len(similar_images)):
            first_file_name = ifn.src.split("/")[-1]
            second_file_name = similar_images[row].src.split("/")[-1]
            confidence_score = images_confidence[row]

            if first_file_name[:7] != second_file_name[:7]:
                list_of_similar_images.append((first_file_name,second_file_name,confidence_score))

    list_of_similar_images = pd.DataFrame(list_of_similar_images,columns=["Master_File","Similar_File","distance_score"])
    list_of_similar_images = list_of_similar_images[list_of_similar_images["distance_score"] >= 0.90]

    return list_of_similar_images

def ensure_directory(directory):
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)