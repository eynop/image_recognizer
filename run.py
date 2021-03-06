import argparse
import os
import time

import recognizer as rec
from recognizer.detector import DETECTORS
from recognizer.matcher import MATCHERS

import cv2


RESULTS_DIR = 'results'


def _get_result_img_path():
    fname = '{}_{}.jpg'.format('result', time.time())
    path = os.path.join(RESULTS_DIR, fname)
    return path


def main(agent):

    if not os.path.exists(RESULTS_DIR):
        os.mkdir(RESULTS_DIR)

    while True:
        try:
            res = agent.recognize()
            cv2.imwrite(_get_result_img_path(), res)
        except Exception as e:
            print(e)


if __name__ == '__main__':
    aparse = argparse.ArgumentParser()
    aparse.add_argument('-iw', '--width', dest='width', default=320,
            help='width to which resize images')
    aparse.add_argument('-ih', '--height', dest='height', default=240,
            help='height to which resize images')
    aparse.add_argument('-d', '--detector', dest='detector', choices=DETECTORS,
            default=DETECTORS[3], help='detector for a tracker')
    aparse.add_argument('-m', '--matcher', dest='matcher', choices=MATCHERS,
            default=MATCHERS[2], help='matcher for a tracker')
    aparse.add_argument('-c', '--camera', dest='camera', type=int,
            default=0, help='camera from which recognize object')
    aparse.add_argument('-i', '--item', dest='item', required=True,
            help='item to recognize on camera stream')

    args = aparse.parse_args()

    # initialize tracker
    tb = rec.tracker.TrackerBuilder()
    tracker = tb.get_tracker(args.detector, args.matcher)

    # initialize camera
    cam = rec.image.Camera(args.camera) 

    # initialize item
    item = rec.image.Item(args.item)

    # create image preprocessor
    preprocessor = rec.image.Preprocessor(args.width, args.height)

    # initialize agent
    agent = rec.agent.Agent(tracker, preprocessor, cam, item)

    main(agent)

