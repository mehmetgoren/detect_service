from abc import ABC, abstractmethod
from typing import List
import numpy as np
import cv2

from core.models.object_detector_model import DetectionBox, BaseDetectedObject
from core.object_detectors.object_detector import ObjectDetector


class ObjectFramerBase(ABC):
    @abstractmethod
    def frame(self, object_detector: ObjectDetector, img: np.array, detected_by: str) -> List[BaseDetectedObject]:
        pass


class DrawObjectFramer(ObjectFramerBase):
    def frame(self, object_detector: ObjectDetector, img: np.array, detected_by: str) -> List[BaseDetectedObject]:
        dis = []
        boxes: List[DetectionBox] = object_detector.get_detect_boxes(img, detected_by)
        for box in boxes:
            di = object_detector.create_detected_object(img, detected_by, box)
            di.detected_by = detected_by
            dis.append(di)
            color = di.get_pred_color()
            xy1 = (box.x1, box.y1)
            xy2 = (box.x2, box.y2)
            cv2.rectangle(img, xy1, xy2, color)
            text = di.get_text()
            cv2.putText(img, text, xy1, cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness=1)
        return dis


class CropObjectFramer(ObjectFramerBase):
    def frame(self, object_detector: ObjectDetector, img: np.array, detected_by: str) -> List[BaseDetectedObject]:
        dis = []
        boxes: List[DetectionBox] = object_detector.get_detect_boxes(img, detected_by)
        for i in range(len(boxes)):
            box = boxes[i]
            sub_img = img[box.y1:box.y2, box.x1:box.x2]
            di = object_detector.create_detected_object(sub_img, detected_by, box)
            di.detected_by = detected_by
            dis.append(di)
        return dis
