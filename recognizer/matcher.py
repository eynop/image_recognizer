import cv2


MATCHERS = ['FLANN', 'BF', 'BFL2', 'BFH']


class MatcherNotFound(Exception):
    pass


class Matcher:

    MIN_MATCH = 10

    def __repr__(self):
        return type(self).__name__

    def match(self, qdes, tdes):
        pass


class FlannMatcher(Matcher):

    def __init__(self):
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 100) # default 30
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def match(self, qdes, tdes):
        matches = self.matcher.knnMatch(qdes, tdes, k = 2)
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
        return good


class BfMatcher(Matcher):

    BF_CROSS_CHECK = False

    def __init__(self):
        self.matcher = cv2.BFMatcher_create(crossCheck = self.BF_CROSS_CHECK)

    def match(self, qdes, tdes):
        if self.BF_CROSS_CHECK:
            return self.matcher.knnMatch(qdes, tdes, k = 2)
        matches = self.matcher.knnMatch(qdes, tdes, k = 2)
        good = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good.append(m)
        return good


class BfL2Matcher(BfMatcher):

    def __init__(self):
        self.matcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck = self.BF_CROSS_CHECK)


class BfHammingMatcher(BfMatcher):

    def __init__(self):
        self.matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck = self.BF_CROSS_CHECK)


class MatcherFactory:

    _MATCHERS = [FlannMatcher, BfMatcher, BfL2Matcher, BfHammingMatcher]

    def get_matcher(self, name):
        n = name.upper()

        for i in range(len(MATCHERS)):
            if n == MATCHERS[i]: return MatcherFactory._MATCHERS[i]()

        raise MatcherNotFound('{} matcher wasn\'t found'.format(name))

