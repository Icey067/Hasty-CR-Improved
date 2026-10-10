"""Skip redundant upstream unit inference when YOLO supplies the board."""


class EmptyUnitDetector:
    def run(self, image):
        return [], []


def configure_unit_detection(detector, vision):
    """Keep upstream inference intact unless the replacement loaded successfully."""
    if vision is not None and vision.ready:
        detector.unit_detector = EmptyUnitDetector()
