from .image import Preprocessor

import cv2


class Agent:

    def __init__(self, tracker, env, item=None):
        self._tracker = tracker
        self._env = env
        self._item = item


    def __repr__(self):
        return 'Agent: Image item recognition agent'


    def set_item(item):
        self._item = item


    def run(self):
        preprocessor = Preprocessor(320, 240)

        #state = self._env.get_state()
        env = preprocessor.process(self._env)
        item = preprocessor.process(self._item)

        if self._item is None:
            print('Agent doesn\'t have item to recognize')
        else: 
            fit_kp, fit_des = self._tracker.detect_and_compute(item)
            query_kp, query_des = self._tracker.detect_and_compute(env)
            matches = self._tracker.match(fit_des, query_des)
            if len(matches) > self._tracker.matcher.MIN_MATCH_COUNT:
                return cv2.drawMatches(env, fit_kp, item, query_kp, matches,None)
            else:
                return env


