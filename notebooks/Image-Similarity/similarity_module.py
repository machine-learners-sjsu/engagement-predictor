from __future__ import unicode_literals
from __future__ import print_function
import youtube_dl
import requests
import re
import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import time
import glob
import os.path
import json
import pandas as pd
import shutil

# Annoy and Scipy for similarity calculation
from annoy import AnnoyIndex
from scipy import spatial


def load_img(path):
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
    i = 0
    start_time = time.time()

    print("---------------------------------")
    print("Step.1 of 2 - mobilenet_v2_140_224 - Loading Started at %s" % time.ctime())
    print("---------------------------------")

    # Definition of module with using tfhub.dev handle
    module_handle = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/5"

    # Load the module
    module = hub.load(module_handle)

    print("---------------------------------")
    print("Step.1 of 2 - mobilenet_v2_140_224 - Loading Completed at %s" % time.ctime())
    print("--- %.2f minutes passed ---------" % ((time.time() - start_time) / 60))

    print("---------------------------------")
    print("Step.2 of 2 - Generating Feature Vectors -  Started at %s" % time.ctime())

    # Loops through all images in a local folder
    for filename in glob.glob('temp_images/*.jpeg'):  # assuming gif
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
        out_path = os.path.join('temp_images/', outfile_name)

        # Saves the 'feature_set' to a text file
        np.savetxt(out_path, feature_set, delimiter=',')

        print("Image feature vector saved to   :%s" % out_path)

    print("---------------------------------")
    print("Step.2 of 2 - Generating Feature Vectors - Completed at %s" % time.ctime())
    print("--- %.2f minutes passed ---------" % ((time.time() - start_time) / 60))
    print("--- %s images processed ---------" % i)


def cluster():
    start_time = time.time()

    print("---------------------------------")
    print("Step.1 - ANNOY index generation - Started at %s" % time.ctime())
    print("---------------------------------")

    # Defining data structures as empty dict
    file_index_to_file_name = {}
    file_index_to_file_vector = {}
    file_index_to_product_id = {}

    # Configuring annoy parameters
    dims = 1792
    n_nearest_neighbors = 20
    trees = 10000

    # Reads all file names which stores feature vectors
    allfiles = glob.glob('temp_images/*.npz')

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

    print("--- Prosess completed in %.2f minutes ---------" % ((time.time() - start_time) / 60))

    # shutil.rmtree("temp_images/", ignore_errors=True)

    return named_nearest_neighbors


def compare_two_video_thumbnails():
    get_image_feature_vectors()
    similar_images_file = cluster()

    similarity_list = []

    for records in similar_images_file:
        similarity_list.append((records['similarity'], records['master_pi'], records['similar_pi']))

    frame = pd.DataFrame(similarity_list, columns=["Similarity", "Master_File", "Similar_File"])
    frame.drop_duplicates(keep="first", inplace=True)
    frame = frame[frame["Similarity"] != 1.0]
    frame.reset_index(drop=True, inplace=True)

    index_to_keep = []
    already_processed_files = []

    for index, row in frame.iterrows():
        file_name_1 = row["Master_File"]
        file_name_2 = row["Similar_File"]

        if (file_name_1, file_name_2) and (file_name_2, file_name_1) not in already_processed_files:
            index_to_keep.append(index)
            already_processed_files.append((file_name_1, file_name_2))
            already_processed_files.append((file_name_2, file_name_1))

    frame = frame.loc[index_to_keep, :]
    frame.to_excel("Similarity.xlsx", index=False)

    return frame


print(compare_two_video_thumbnails())
