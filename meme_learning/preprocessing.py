from io import BytesIO
from PIL import Image
from imagenet import classify_image

import os
import glob

import numpy as np
import scipy.ndimage as spimg

JPEG_FORMAT = "JPEG"
WILD_PNG = "*.png"
WILD_JPG = "*.jpg"
CLASSIFIER_IMG_SIZE = (299, 299)

DATA_DIR = "data/"
TRAIN_DIR = "train/"

objects_detected = {}

class DataUnit:
    def __init__(self, image_path):
        global objects_detected

        tmp = image_path[ len(TRAIN_DIR) : -4 ].split("_")

        self.rating = int(tmp[1])
        self.ID = "_".join(tmp[2:])

        self.mat = spimg.imread(image_path)

        # Split the channels into different segments of the image with some
        # padding
        self.mat = np.r_[
            self.mat[:,:,0],
            np.zeros((2, 299,)),
            self.mat[:,:,1],
            np.zeros((2, 299,)),
            self.mat[:,:,2]
        ]
        assert self.mat.shape == (299 * 3 + 4, 299)

        # Add an extra column to pad to (299*3 + 4) x 300
        self.mat = np.c_[ self.mat, np.zeros(299 * 3 + 4) ]

        objects_extracted = classify_image.classify(image_path,
                print_results=False)

        self.objects = {}
        for extracted in objects_extracted:
            # ID of object is the shortest one
            ids = extracted[0].split(", ")
            ids.sort(key=len)

            obj_id = ids[0].lower()
            self.objects[obj_id] = extracted[1]

            if obj_id not in objects_detected:
                objects_detected[obj_id] = len(objects_detected)

        self.obj_vec = None

def scale_images(src_path, dst_path):
    original_img = Image.open(src_path)
    resized = original_img.resize(CLASSIFIER_IMG_SIZE, Image.ANTIALIAS)

    # Save the resized image to destination
    resized.save(dst_path, format=JPEG_FORMAT)

def process_for_training(images):
    processed_paths = []
    for image in images:
        basename = image[ len(DATA_DIR) : -4 ]
        print "Resizing %s and converting to JPEG..." % (basename,)
        processed_path = os.path.join(TRAIN_DIR, basename + ".jpg")
        scale_images(image, processed_path)
        processed_paths.append(processed_path)
    return processed_path

def get_training_set():
    global objects_detected

    raw = glob.glob(os.path.join(DATA_DIR, WILD_PNG))
    raw += glob.glob(os.path.join(DATA_DIR, WILD_JPG))
    images = process_for_training(raw)

    dataunits = []
    for image in images:
        print "Image: %s (%d of %d)" % (image, len(dataunits) + 1, len(images))
        dataunits.append(DataUnit(image))

    print "%d unique objects detected." % (len(objects_detected),)

    one_hot_size = ((len(objects_detected) // 300) + 2) * 300
    for dataunit in dataunits:
        dataunit.obj_vec = np.zeros( (one_hot_size,), dtype=np.float32 )
        for obj_id in dataunit.objects:
            dataunit.obj_vec[objects_detected[obj_id]] = dataunit.objects[obj_id]

        mod4 = (dataunit.mat.shape[0] + (one_hot_size // 300)) % 4
        padding = np.zeros((4 - mod4, 300))

        dataunit.mat = np.r_[
            dataunit.mat,
            padding,
            np.reshape(dataunit.obj_vec, (one_hot_size // 300, 300))
        ]

    return np.array([
        dataunit.mat
        for dataunit in dataunits
    ], dtype=np.float32), np.array([
        dataunit.rating
        for dataunit in dataunits
    ], dtype=np.int32)

def get_single_img(img_path):
    processed = process_for_training([img_path])
    dataunit = DataUnit(processed[0])
    return dataunit.mat

if __name__ == "__main__":
    process_for_training()

    A, y = get_training_set()
    print "Final shape of A: %r" % (A.shape,)
    print "Final shape of y: %r" % (y.shape,)

