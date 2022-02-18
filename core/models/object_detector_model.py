from abc import ABC, abstractmethod
import numpy as np
from typing import List

from core.models.detected_objects import BaseDetectedObject


class DetectionBox:
    def __init__(self, x1: int, y1: int, x2: int, y2: int, confidence: float, cls: int):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.confidence = confidence
        self.cls_idx = cls


class BaseObjectDetectorModel(ABC):
    @abstractmethod
    def get_detect_boxes(self, img: np.array, detected_by: str) -> List[DetectionBox]:
        raise NotImplementedError('BaseObjectDetectorModel.get_detect_boxes')

    @abstractmethod
    def create_detected_object(self, img: np.array, detected_by: str, box: DetectionBox) -> BaseDetectedObject:
        raise NotImplementedError('BaseObjectDetectorModel.create_detected_object')

    @abstractmethod
    def get_detected_object_class_name(self, cls_idx: int) -> str:
        raise NotImplementedError('BaseObjectDetectorModel.get_detected_object_class_name')
