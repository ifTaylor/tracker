import cv2

from src.kalman import KalmanFilterTracker


class MotionTracker:
    def __init__(self, detector):
        self.detector = detector
        self.kalman_tracker = KalmanFilterTracker()

    def run(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            x, y = self.detector.detect(frame)
            if x is not None and y is not None:
                predicted_x, predicted_y = self.kalman_tracker.update((x, y))
                if predicted_x is not None and predicted_y is not None:
                    cv2.circle(frame, (int(predicted_x), int(predicted_y)), 5, (0, 255, 0), -1)
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()