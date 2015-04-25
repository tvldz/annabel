#!/usr/bin/env python
# encoding: utf-8
"""
 _ __ __  _ |_  _  |
(_|| || |(_||_)(/_ |

annabel.py
Approximate Nearest Neighbor Assisted Generative Collage
Copyright (C) 2015 Thomas Valadez (@tvldz)
License: GPLv2

Requires
annoy: https://github.com/spotify/annoy
pillow: http://python-pillow.github.io/
"""
import argparse
import os
import sys
from os import listdir
from os.path import isfile, join
from shutil import copyfile
import pickle
from annoy import AnnoyIndex
from PIL import Image

PROFILES_DIRECTORY = "profiles/"
OUTPUT_DIRECTORY = "output/"
INPUT_DIRECTORY = "input_images/"
CROP_HEIGHT = 40
CROP_WIDTH = 40
CROP_INCREMENT = 20
SAMPLE_DIMENSION = 10,10 # 10x10 (100) dimension vector sample
TREE_SIZE = 10  # number of trees to create for ANN search.


def main():
    """
    argv[1] represents 3 commands:
    gather: create a "profile" of source images from which a collage may be formed.
    create: create a new collage given an image and a profile.
    list: list the available profiles.
    """
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_a = subparsers.add_parser(
        "gather", help="create a new source profile")
    parser_a.add_argument("-n", "-name", type=str, action="store",
                          required=True, help="name for the new profile", dest="name")
    parser_a.add_argument("-f", "-folder", type=str, default=INPUT_DIRECTORY,
                          help="path to directory with source images", dest="images_folder")
    parser_a.add_argument(
        "-w", "-width", type=int, default=CROP_WIDTH, help="crop width", dest="cwidth")
    parser_a.add_argument(
        "-j", "-height", type=int, default=CROP_HEIGHT, help="crop height", dest="cheight")
    parser_a.add_argument(
        "-i", "-increment", type=int, default=CROP_INCREMENT, help="crop increment", dest="cincrement")

    parser_b = subparsers.add_parser(
        "create", help="process source images into a profile")
    parser_b.add_argument("-i", "-image", type=str, action="store",
                          required=True, help="path to source image", dest="input_image")
    parser_b.add_argument("-p", "-profile", type=str, action="store",
                          required=True, help="profile to use", dest="profile_name")
    parser_b.add_argument("-c", "-count", type=int, action="store", default=1,
                          help="number of versions to output", dest="version_count")

    parser_c = subparsers.add_parser("list", help="list available profiles")

    results = parser.parse_args()
    if sys.argv[1] == "gather":
        create_profile(results.name, results.images_folder,
                       results.cwidth, results.cheight, results.cincrement)
    elif sys.argv[1] == "create":
        create_collage(
            results.input_image, results.profile_name, results.version_count)
    elif sys.argv[1] == "list":
        get_profiles()
    return


def create_profile(profile_name, image_folder, crop_width, crop_height, crop_increment):
    """
    given a folder and profile name, gather a series of subimages into a profile
    with which to create a collage.
    """
    profile_folder = PROFILES_DIRECTORY + profile_name + "/"
    if not os.path.exists(profile_folder):
        os.makedirs(profile_folder)
    if not os.path.exists(profile_folder + "images/"):
        os.makedirs(profile_folder + "images/")
    image_file_list = [
        f for f in listdir(image_folder) if isfile(join(image_folder, f))]
    # todo: use crop ratio to calculate variable vector size
    nns_index = AnnoyIndex(SAMPLE_DIMENSION[0]*SAMPLE_DIMENSION[1], metric="euclidean")
    image_index = []
    # image_index[0] holds profile metadata.
    image_index.insert(0,{"crop_width": crop_width, "crop_height": crop_height})
    index = 1
    # iterate over images for processing into boxes and associated feature vectors
    for image_file in image_file_list:
        print("processing {}...").format(image_file),
        image_destination = profile_folder + "images/" + image_file
        copyfile(image_folder + image_file, image_destination)
        image = Image.open(image_destination)
        image_width, image_height = image.size[0], image.size[1]
        for x in xrange(0, image_width, crop_increment):
            for y in xrange(0, image_height, crop_increment):
                box = (x, y, x + crop_width, y + crop_height)
                image_sample = image.crop(box).resize(
                    SAMPLE_DIMENSION).convert("LA")  # dimensionality reduction
                gs_pixeldata = []  # reset feature vector
                # create feature vector for annoy
                for pixel in list(image_sample.getdata()):
                    gs_pixeldata.append(pixel[0])
                # add feature vector to annoy
                nns_index.add_item(index, gs_pixeldata)
                image_index.insert(
                    index, {"image": image_destination, "box": (x, y, x + crop_width, y + crop_height)})
                index += 1
        print("done.")
    print("{} total subimages to be indexed...").format(str(index-1))
    print("building trees (this can take awhile)...")
    nns_index.build(TREE_SIZE)  # annoy builds trees
    print("done.")
    print("serializing trees..."),
    nns_index.save(profile_folder + profile_name + ".tree")
    print("done.")
    print("serializing index..."),
    pickle.dump(image_index, open(profile_folder + profile_name + ".p", "wb"))
    print("done.")
    print("{} profile completed. Saved in {}").format(
        profile_name, profile_folder)
    return


def create_collage(input_image, profile_name, version_count):
    """
    given an input image and an existing profile, create a set of new collages.
    """
    profile_folder = PROFILES_DIRECTORY + profile_name + "/"
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)
    # todo: load feature dimensions from profile
    nns_index = AnnoyIndex(SAMPLE_DIMENSION[0]*SAMPLE_DIMENSION[1], metric="euclidean")
    print("loading trees...")
    nns_index.load(profile_folder + profile_name + ".tree")
    print("done.")
    subimage_index = pickle.load(
        open(profile_folder + profile_name + ".p", "rb"))
    template_image = Image.open(input_image)
    image_width, image_height = template_image.size[0], template_image.size[1]
    crop_width, crop_height = subimage_index[0]["crop_width"], subimage_index[0]["crop_height"]
    for i in xrange(version_count):
        print("Creating collage {}/{}...").format(i+1, version_count)
        output_image = template_image.copy()
        for x in xrange(0, image_width, crop_width):
            for y in xrange(0, image_height, crop_height):
                box = (x, y, x + crop_width, y + crop_height)
                crop_box = output_image.crop(box)
                crop_sample = crop_box.convert("LA").resize(SAMPLE_DIMENSION)
                gs_pixeldata = []
                for pixel in list(crop_sample.getdata()):
                    gs_pixeldata.append(pixel[0])
                image_neighbor = nns_index.get_nns_by_vector(
                    gs_pixeldata, version_count)[i]
                substitute_image = Image.open(
                    subimage_index[image_neighbor]["image"])
                substitute_crop = substitute_image.crop(
                    subimage_index[image_neighbor]["box"])
                output_image.paste(substitute_crop, box)
        output_path = OUTPUT_DIRECTORY + str(i) + ".png"
        output_image.save(output_path, "PNG")
        print("done.")
    print("{} image(s) saved in {}").format(
            version_count, OUTPUT_DIRECTORY)
    return


def get_profiles():
    """
    list the available profiles and associated metadata
    """
    print("Available Profiles:")
    for directory in os.listdir(PROFILES_DIRECTORY):
        print(directory)
        # todo: print profile metadata
        return

if __name__ == "__main__":
    sys.exit(main())
