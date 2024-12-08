import numpy as np
import cv2
import os

class SimpleDatasetLoader:
    def __init__(self, preprocessors:list=None):
        # stores the image preprocessor
        self.preprocessors = preprocessors

        # if no preprocessors are supplied, initialize them as
        # an empty list
        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, image_paths, verbose=-1):
        # initializes the list of features ane labels
        data = []
        labels = []

        # loop over the input images
        for (i, image_path) in enumerate(image_paths):
            # load the image and extract the class label assuming
            # the image paths has the following format:
            # /path/to/dataset/{class}/{image}.jpg
            image = cv2.imread(image_path)
            label = image_path.split(os.path.sep)[-2]

            # data.append(p.preprocess(image) for p in self.preprocessors if self.preprocessors is not None) inefficient

            # loop over preprocessors(if available) and preprocess
            # the already loaded images
            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocess(image)
            
            data.append(image)
            labels.append(label)

            # shows and update for every `verbose` iamges
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print(f"[INFO]: processed {i + 1}/{len(image_paths)}")

        # return tuple of labels and their respective preprocessed images
        return (np.array(data), np.array(labels))