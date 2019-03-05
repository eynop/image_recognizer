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

