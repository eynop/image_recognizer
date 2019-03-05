import cv2


DETECTORS = [ 'SIFT', 'SURF', 'KAZE', 'AKAZE', 'BRISK', 'ORB' ]


class DetectorNotFound(Exception):
    pass


class Detector:
    '''Base class for detector wrappers'''

    def __init__(self, detector):
        self.detector = detector


    def __repr__(self):
        return type(self).__name__


    def detect_and_compute(self, img):
        kp, des = self.detector.detectAndCompute(img, None)
        return kp, des


class SiftDetector(Detector):
    '''Wrapper for non-free SIFT detector'''

    def __init__(self):
        self.detector = cv2.xfeatures2d.SIFT_create()


class SurfDetector(Detector):
    '''Wrapper for non-free SURF detector'''

    def __init__(self):
        self.detector = cv2.xfeatures2d.SURF_create()


class KazeDetector(Detector):
    '''Wrapper for free KAZE detector'''

    def __init__(self):
        self.detector = cv2.KAZE_create()


class AkazeDetector(Detector):
    '''Wrapper for free AKAZE detector'''

    def __init__(self):
        self.detector = cv2.AKAZE_create()


class BriskDetector(Detector):
    '''Wrapper for free BRISK detector'''

    def __init__(self):
        self.detector = cv2.BRISK_create()


class OrbDetector(Detector):
    '''Wrapper for free ORB detector'''

    def __init__(self):
        self.detector = cv2.ORB_create()


class DetectorFactory:
    
    _DETECTORS = [SiftDetector, SurfDetector, KazeDetector,
                AkazeDetector, BriskDetector, OrbDetector]

    def __repr__(self):
        return '{}: {}'.format(type(self).__name__, DetectorFactory._DETECTORS)


    def get_detector(self, name):
        n = name.upper()

        for i in range(len(DetectorFactory._DETECTORS)):
            if n == DETECTORS[i]: return DetectorFactory._DETECTORS[i]()

        raise DetectorNotFound('{} detector wasn\'t found'.format(name))

