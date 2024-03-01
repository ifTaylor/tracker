import cv2

from src.detector import Detector


class ObjectDetector(Detector):
    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2()

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = self.bg_subtractor.apply(gray)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            return x + w // 2, y + h // 2
        else:
            return None, None
