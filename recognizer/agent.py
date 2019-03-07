from .image import Preprocessor

import cv2
import numpy as np


class Agent:

    def __init__(self, tracker, preprocessor, env, item=None):
        self._tracker = tracker
        self._preprocessor = preprocessor
        self._env = env
        self.item = item


    def __repr__(self):
        return '{}: Image item recognition agent'.format(type(self).__name__)


    def _draw_detected(self, matches, dquery_kp, dtrain_kp, dquery, dtrain, train):
        src_pts = np.float32([dquery_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([dtrain_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        h, w = dquery.shape[:2]
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)

        dst = cv2.perspectiveTransform(pts, M)

        # rescale
        # stupid, but works
        for i in range(dst.shape[0]):
            y = np.interp(dst.item(i, 0, 0), [0, dtrain.shape[0]], [0, train.shape[0]])
            x = np.interp(dst.item(i, 0, 1), [0, dtrain.shape[1]], [0, train.shape[1]])
            dst.itemset((i, 0, 1), x)
            dst.itemset((i, 0, 0), y)

        img = cv2.polylines(train, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        
        return img


    def set_item(item):
        self.item = item


    def run(self):
        while True:
            self.recognize()


    def recognize(self):
        if self.item is None:
            print('Agent doesn\'t have item to recognize')
        else:
            # train image is enviroment state
            train = self._env.get_img()
            dtrain = self._preprocessor.process(train) # resize, grayscale
            dt_kp, dt_des = self._tracker.detect_and_compute(dtrain)

            # query image is image which will be searched for
            query = self.item.get_img()
            dquery = self._preprocessor.process(query) # resize, gayscale
            if self.item.kp is None:
                dq_kp, dq_des = self._tracker.detect_and_compute(dquery)
                self.item.kp = dq_kp
                self.item.des = dq_des
            else:
                dq_kp = self.item.kp
                dq_des = self.item.des
           
            matches = self._tracker.match(dq_des, dt_des)

            if len(matches) >= self._tracker.matcher.MIN_MATCH:
                print('Matches found: {}/{}'.format(len(matches), self._tracker.matcher.MIN_MATCH))
                img = self._draw_detected(matches, dq_kp, dt_kp, dquery, dtrain, train)
                return img
            else:
                print('Not enough matches: {}/{}'.format(len(matches), self._tracker.matcher.MIN_MATCH))
                return train

