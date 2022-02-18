import numpy as np
from typing import List

from core.models.detected_objects import BaseDetectedObject, Coco80DetectedObject, coco80_info
from core.models.object_detector_model import BaseObjectDetectorModel, DetectionBox


class Yolov5ObjectDetectorModel(BaseObjectDetectorModel):
    def __init__(self, model):
        super(Yolov5ObjectDetectorModel, self).__init__()
        self.model = model

    def get_detect_boxes(self, img: np.array, detected_by: str) -> List[DetectionBox]:
        detections = self.model(img)
        # print img1 predictions (pixels)
        #                   x1           y1           x2           y2   confidence        class
        # tensor([[7.50637e+02, 4.37279e+01, 1.15887e+03, 7.08682e+02, 8.18137e-01, 0.00000e+00],
        #         [9.33597e+01, 2.07387e+02, 1.04737e+03, 7.10224e+02, 5.78011e-01, 0.00000e+00],
        #         [4.24503e+02, 4.29092e+02, 5.16300e+02, 7.16425e+02, 5.68713e-01, 2.70000e+01]])
        # Output will be a numpy array in the following format:
        # [[x1, y1, x2, y2, confidence, class]]
        boxes: List[DetectionBox] = []
        for box in detections.xyxy[0]:
            x1, y1, x2, y2, conf, cls = int(box[0].item()), int(box[1].item()), int(box[2].item()), int(box[3].item()), box[4].item(), int(
                box[5].item())
            boxes.append(DetectionBox(x1, y1, x2, y2, conf, cls))
        return boxes

    def create_detected_object(self, img: np.array, detected_by: str, box: DetectionBox) -> BaseDetectedObject:
        obj = Coco80DetectedObject(img, box.confidence, box.cls_idx)
        obj.detected_by = detected_by
        return obj

    def get_detected_object_class_name(self, cls_idx: int) -> str:
        return coco80_info.get_name(cls_idx)
