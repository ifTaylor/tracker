from src.tracker import MotionTracker
from src.object_detector import ObjectDetector
from src.color_detector import ObjectColorDetector


if __name__ == "__main__":
    track_color = True
    color = 'yellow'

    if track_color:
        tracker = MotionTracker(ObjectColorDetector(color))
    else:
        tracker = MotionTracker(ObjectDetector())
    tracker.run()
