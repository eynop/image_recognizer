import cv2
import recognizer as rec


preprocessor = rec.image.Preprocessor(320, 240)
img = cv2.imread('self.JPG')
print(img.shape)
new_img = preprocessor.process(img)
print(new_img.shape)
print(img.shape)
cv2.imwrite('grey.jpg', new_img)

tb = rec.tracker.TrackerBuilder()
tracker = tb.get_tracker('akaze', 'bfh')

agent = rec.agent.Agent(tracker, img, img)  
result = agent.run()
cv2.imwrite('result.jpg', result)
