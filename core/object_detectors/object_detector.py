from abc import abstractmethod
from typing import List
import numpy as np

from core.models.detected_objects import BaseDetectedObject
from core.models.object_detector_model import BaseObjectDetectorModel, DetectionBox


class ObjectDetector(BaseObjectDetectorModel):
    def __init__(self, detector_model: BaseObjectDetectorModel):
        super(ObjectDetector, self).__init__()
        self.concrete = detector_model

    @abstractmethod
    def get_detect_boxes(self, img: np.array, detected_by: str) -> List[DetectionBox]:
        raise NotImplementedError('ObjectDetector.get_detect_boxes')

    @abstractmethod
    def create_detected_object(self, img: np.array, detected_by: str, box: DetectionBox) -> BaseDetectedObject:
        raise NotImplementedError('ObjectDetector.create_detected_object')

    def get_detected_object_class_name(self, cls_idx: int) -> str:
        return self.concrete.get_detected_object_class_name(cls_idx)
