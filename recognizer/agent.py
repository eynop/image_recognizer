from .image import Preprocessor

import cv2


class Agent:

    def __init__(self, tracker, preprocessor, env, item=None):
        self._tracker = tracker
        self._preprocessor = preprocessor
        self._env = env
        self.item = item


    def __repr__(self):
        return '{}: Image item recognition agent'.format(type(self).__name__)


    def set_item(item):
        self.item = item


    def run(self):
        while True:
            self.recognize()


    def recognize(self):
        if self.item is None:
            print('Agent doesn\'t have item to recognize')
        else:
            train = self.item.get_img()
            dtrain = self._preprocessor.process(train)
            if self.item.kp is None:
                t_kp, t_des = self._tracker.detect_and_compute(dtrain)
                self.item.kp = t_kp
                self.item.des = t_des
            else:
                t_kp = self.item.kp
                t_des = self.item.des
            
            query = self._env.get_img()
            dquery = self._preprocessor.process(query)
            q_kp, q_des = self._tracker.detect_and_compute(dquery)
           
            matches = self._tracker.match(t_des, q_des)

            if len(matches) > self._tracker.matcher.MIN_MATCH_COUNT:
                return cv2.drawMatches(dtrain, t_kp, dquery, q_kp, matches, None)
            else:
                print('not enough matches')
                return query

