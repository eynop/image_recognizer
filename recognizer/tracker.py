try:
    from .detector import DetectorFactory, DetectorNotFound
except (SystemError, ImportError):
    from detector import DetectorFactory, DetectorNotFound

try:
    from .matcher import MatcherFactory, MatcherNotFound
except (SystemError, ImportError):
    from matcher import MatcherFactory, MatcherNotFound


class TrackerBuilderError(Exception):
    pass


class TrackerBuilder:

    def __init__(self):
        self.mfactory = MatcherFactory()
        self.dfactory = DetectorFactory()


    def get_tracker(self, dname, mname):

        try:
            detector = self.dfactory.get_detector(dname)
            matcher = self.mfactory.get_matcher(mname)
            return Tracker(detector, matcher)
        except (DetectorNotFound, MatcherNotFound) as e:
            raise TrackerBuilderError(
                    'Couldn\'t build tracker with {} detector and {} matcher'
                    .format(dname, mname)
                    )


class Tracker:
    
    def __init__(self, detector, matcher):
        self.detector = detector
        self.matcher = matcher


    def __repr__(self):
        return 'Tracker: {}'.format(self.get_name())


    def get_name(self):
        return '{} {}'.format(
                str(self.detector),
                str(self.matcher)
                )


    def detect_and_compute(self, img):
        kp, des = self.detector.detect_and_compute(img)
        return kp, des
    

    def match(self, qdes, tdes):
        matches = self.matcher.match(qdes, tdes)
        return matches

