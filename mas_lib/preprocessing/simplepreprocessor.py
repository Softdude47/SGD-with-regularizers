import cv2


class SimplePreprocessor:
    def __init__(self, width:int, height:int, inter=cv2.INTER_AREA):
        # set target image width, height and interpolation function
        # for resizing

        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        # resizes the image ignoring the aspect ratio
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)
