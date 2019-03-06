import os
import cv2


class Preprocessor:

    def __init__(self, width, height, grey=True):
        self._width = width
        self._height = height
        self._grey = grey

    
    def process(self, img):
        height, width = img.shape[:2]

        img_ = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if self._grey else img

        scale = self._width / width if width > height else self._height / height

        nheight, nwidth = int(height * scale), int(width * scale)

        if scale < 1: # shrink
            return cv2.resize(img_, (nwidth, nheight), cv2.INTER_AREA)
        # zoom
        return cv2.resize(img_, (nwidth, nheight), cv2.INTER_CUBIC)


class Camera:

    def __init__(self, idx):
        self._idx = idx
        self._cap = cv2.VideoCapture(idx)

    
    def __repr__(self):
        return '{} idx {}'.format(type(self).__name__, self._idx)


    def _get_last_frame(self):
        bs = self._cap.get(cv2.CAP_PROP_BUFFERSIZE)
        while bs >= 0:
            ret, frame = self._cap.read()
            bs -= 1
        return frame


    def get_img(self):
        return self._get_last_frame()


class Item:

    def __init__(self, path):
        self._img = cv2.imread(path)
        self._path = os.path.abspath(path)
        self.kp = None
        self.des = None


    def __repr__(self):
        return '{} image: {}'.format(type(self).__name__, self._path)
    

    def get_img(self):
        return self._img

