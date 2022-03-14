from __future__ import print_function
import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import time
import glob
import os.path
import json

# Standard PySceneDetect imports:
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.scene_manager import generate_images
# For caching detection metrics and saving/loading to a stats file
from scenedetect.stats_manager import StatsManager

# For content-aware scene detection:
from scenedetect.detectors.content_detector import ContentDetector

# Annoy and Scipy for similarity calculation
from annoy import AnnoyIndex
from scipy import spatial


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

        print('List of scenes obtained:')
        for i, scene in enumerate(scene_list):
            print(
                'Scene %2d: Start %s / Frame %d, End %s / Frame %d' % (
                    i + 1,
                    scene[0].get_timecode(), scene[0].get_frames(),
                    scene[1].get_timecode(), scene[1].get_frames(),))

        # We only write to the stats file if a save is required:
        if stats_manager.is_save_required():
            with open(stats_file_path, 'w') as stats_file:
                stats_manager.save_to_csv(stats_file, base_timecode)

        video_file_name = video_path.split(".")[-2]
        video_file_process_path = "temp_video_frames/"+video_file_name
        generate_images(scene_list, video_manager, video_file_process_path, num_images=1)

    finally:
        video_manager.release()

    return scene_list


def load_img(path):
    """
    #################################################
    # This function:
    # Loads the JPEG image at the given path
    # Decodes the JPEG image to a uint8 W X H X 3 tensor
    # Resizes the image to 224 x 224 x 3 tensor
    # Returns the pre processed image as 224 x 224 x 3 tensor
    #################################################
    """

    # Reads the image file and returns data type of string
    img = tf.io.read_file(path)

    # Decodes the image to W x H x 3 shape tensor with type of uint8
    img = tf.io.decode_jpeg(img, channels=3)

    # Resize the image to 224 x 244 x 3 shape tensor
    img = tf.image.resize_with_pad(img, 224, 224)

    # Converts the data type of uint8 to float32 by adding a new axis
    # This makes the img 1 x 224 x 224 x 3 tensor with the data type of float32
    # This is required for the mobilenet model we are using
    img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]

    return img


def get_image_feature_vectors():
    """
    # This function:
    # Loads the mobilenet model in TF.HUB
    # Makes an inference for all images stored in a local folder
    # Saves each of the feature vectors in a file
    """
    i = 0

    start_time = time.time()

    print("---------------------------------")
    print("Step.1 of 2 - mobilenet_v2_140_224 - Loading Started at %s" % time.ctime())
    print("---------------------------------")

    # Definition of module with using tfhub.dev handle
    # module_handle = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4"
    # module_handle = "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4"
    # module_handle = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
    # module_handle = "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4"

    # Definition of module with using tfhub.dev handle
    module_handle = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
    module = hub.load(module_handle)

    print("---------------------------------")
    print("Step.1 of 2 - mobilenet_v2_140_224 - Loading Completed at %s" % time.ctime())
    print("--- %.2f minutes passed ---------" % ((time.time() - start_time) / 60))

    print("---------------------------------")
    print("Step.2 of 2 - Generating Feature Vectors -  Started at %s" % time.ctime())

    # Loops through all images in a local folder
    for filename in glob.glob('temp_video_frames/video_process_workplace/*.jpg'):  # assuming gif
        i = i + 1

        print("-----------------------------------------------------------------------------------------")
        print("Image count                     :%s" % i)

        # Loads and pre-process the image
        img = load_img(filename)

        # Calculate the image feature vector of the img
        features = module(img)

        # Remove single-dimensional entries from the 'features' array
        feature_set = np.squeeze(features)

        # Saves the image feature vectors into a file for later use

        outfile_name = os.path.basename(filename).split('.')[0] + ".npz"
        out_path = os.path.join('temp_video_frames/video_process_workplace/', outfile_name)

        # Saves the 'feature_set' to a text file
        np.savetxt(out_path, feature_set, delimiter=',')

        print("Image feature vector saved to   :%s" % out_path)

    print("---------------------------------")
    print("Step.2 of 2 - Generating Feature Vectors - Completed at %s" % time.ctime())
    print("--- %.2f minutes passed ---------" % ((time.time() - start_time) / 60))
    print("--- %s images processed ---------" % i)


def cluster():
    #################################################
    # This function;
    # Reads all image feature vectores stored in /feature-vectors/*.npz
    # Adds them all in Annoy Index
    # Builds ANNOY index
    # Calculates the nearest neighbors and image similarity metrics
    # Stores image similarity scores with productID in a json file
    #################################################

    start_time = time.time()

    print("---------------------------------")
    print("Step.1 - ANNOY index generation - Started at %s" % time.ctime())
    print("---------------------------------")

    # Defining data structures as empty dict
    file_index_to_file_name = {}
    file_index_to_file_vector = {}
    file_index_to_product_id = {}

    # Configuring annoy parameters
    # dims = 1792
    # dims = 2048
    # dims = 1280
    dims = 1280
    n_nearest_neighbors = 20
    trees = 10000

    # Reads all file names which stores feature vectors
    allfiles = glob.glob('temp_video_frames/video_process_workplace/*.npz')

    t = AnnoyIndex(dims, metric='angular')

    for file_index, i in enumerate(allfiles):
        # Reads feature vectors and assigns them into the file_vector
        file_vector = np.loadtxt(i)

        # Assigns file_name, feature_vectors and corresponding product_id
        file_name = os.path.basename(i).split('.')[0]
        file_index_to_file_name[file_index] = file_name
        file_index_to_file_vector[file_index] = file_vector
        file_index_to_product_id[file_index] = file_name

        # Adds image feature vectors into annoy index
        t.add_item(file_index, file_vector)

        print("---------------------------------")
        print("Annoy index     : %s" % file_index)
        print("Image file name : %s" % file_name)
        print("Product id      : %s" % file_index_to_product_id[file_index])
        print("--- %.2f minutes passed ---------" % ((time.time() - start_time) / 60))

    # Builds annoy index
    t.build(trees)

    print("Step.1 - ANNOY index generation - Finished")
    print("Step.2 - Similarity score calculation - Started ")

    named_nearest_neighbors = []

    # Loops through all indexed items
    for i in file_index_to_file_name.keys():

        # Assigns master file_name, image feature vectors and product id values
        master_file_name = file_index_to_file_name[i]
        master_vector = file_index_to_file_vector[i]
        master_product_id = file_index_to_product_id[i]

        # Calculates the nearest neighbors of the master item
        nearest_neighbors = t.get_nns_by_item(i, n_nearest_neighbors)

        # Loops through the nearest neighbors of the master item
        for j in nearest_neighbors:
            print(j)

            # Assigns file_name, image feature vectors and product id values of the similar item
            neighbor_file_name = file_index_to_file_name[j]
            neighbor_file_vector = file_index_to_file_vector[j]
            neighbor_product_id = file_index_to_product_id[j]

            # Calculates the similarity score of the similar item
            similarity = 1 - spatial.distance.cosine(master_vector, neighbor_file_vector)
            rounded_similarity = int((similarity * 10000)) / 10000.0

            # Appends master product id with the similarity score
            # and the product id of the similar items
            named_nearest_neighbors.append({
                'similarity': rounded_similarity,
                'master_pi': str(master_product_id),
                'similar_pi': str(neighbor_product_id)})

        print("---------------------------------")
        print("Similarity index       : %s" % i)
        print("Master Image file name : %s" % file_index_to_file_name[i])
        print("Nearest Neighbors.     : %s" % nearest_neighbors)
        print("--- %.2f minutes passed ---------" % ((time.time() - start_time) / 60))

    print("Step.2 - Similarity score calculation - Finished ")
    print("--- Process completed in %.2f minutes ---------" % ((time.time() - start_time) / 60))

    return named_nearest_neighbors

