import cv2
import numpy as np

from src.detector import Detector


class ObjectColorDetector(Detector):
    def __init__(self, color='yellow'):
        self.color = color

    def detect(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        bgr_lower, bgr_upper = self.get_color_bounds(self.color)
        mask = cv2.inRange(hsv, bgr_lower, bgr_upper)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            return x + w // 2, y + h // 2
        else:
            return None, None

    def get_color_bounds(self, color):
        lower_bounds = {
            'yellow': np.array([20, 100, 100]),
            'green': np.array([40, 100, 100]),
            'blue': np.array([90, 100, 100]),
            'red': np.array([0, 100, 100]),
            'purple': np.array([130, 100, 100]),
            'orange': np.array([10, 100, 100]),
            'pink': np.array([150, 100, 100]),
            'cyan': np.array([70, 100, 100]),
            'brown': np.array([5, 50, 50]),
            'white': np.array([0, 0, 200]),
            'black': np.array([0, 0, 0]),
            'gray': np.array([0, 0, 100]),
            'magenta': np.array([140, 100, 100]),
            'turquoise': np.array([100, 100, 100]),
            'lavender': np.array([120, 50, 50]),
            'peach': np.array([5, 100, 100]),
            'olive': np.array([40, 100, 50]),
        }

        upper_bounds = {
            'yellow': np.array([30, 255, 255]),
            'green': np.array([80, 255, 255]),
            'blue': np.array([120, 255, 255]),
            'red': np.array([10, 255, 255]),
            'purple': np.array([160, 255, 255]),
            'orange': np.array([20, 255, 255]),
            'pink': np.array([170, 255, 255]),
            'cyan': np.array([100, 255, 255]),
            'brown': np.array([20, 255, 200]),
            'white': np.array([255, 50, 255]),
            'black': np.array([179, 255, 40]),
            'gray': np.array([179, 50, 150]),
            'magenta': np.array([160, 255, 255]),
            'turquoise': np.array([120, 255, 255]),
            'lavender': np.array([140, 255, 200]),
            'peach': np.array([20, 255, 255]),
            'olive': np.array([80, 255, 150]),
        }

        return lower_bounds[color], upper_bounds[color]
