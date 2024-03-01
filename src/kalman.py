import cv2
import numpy as np


class KalmanFilterTracker:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                                 [0, 1, 0, 0],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32) * 0.03
        self.last_measurement = None
        self.last_prediction = None

    def update(self, measurement):
        if measurement is None:
            return None, None

        measurement = np.array(measurement, dtype=np.float32)

        if self.last_measurement is None:
            # Initialize
            self.kalman.statePre = np.array([[measurement[0]], [measurement[1]], [0], [0]], dtype=np.float32)
            self.kalman.statePost = np.array([[measurement[0]], [measurement[1]], [0], [0]], dtype=np.float32)
            self.last_measurement = measurement
            self.last_prediction = measurement
            return measurement[0], measurement[1]
        else:
            # Prediction
            prediction = self.kalman.predict()
            self.last_prediction = prediction

            # Correction
            self.kalman.correct(measurement)
            self.last_measurement = measurement

        return prediction[0][0], prediction[1][0]