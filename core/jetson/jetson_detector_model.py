import numpy as np
from typing import List
import jetson.utils as utils

from core.models.detected_objects import BaseDetectedObject, Coco91DetectedObject
from core.models.object_detector_model import BaseObjectDetectorModel, DetectionBox


class JetsonbjectDetectorModel(BaseObjectDetectorModel):
    def __init__(self, net):
        super(JetsonbjectDetectorModel, self).__init__()
        self.net = net

    def get_detect_boxes(self, img: np.array, detected_by: str) -> List[DetectionBox]:
        img_cuda = utils.cudaFromNumpy(img)
        detections = self.net.Detect(img_cuda, overlay='OVERLAY_NONE')

        boxes: List[DetectionBox] = []
        for d in detections:
            x1, y1, x2, y2 = int(d.Left), int(d.Top), int(d.Right), int(d.Bottom)
            conf, cls = d.Confidence, d.ClassID
            boxes.append(DetectionBox(x1, y1, x2, y2, conf, cls))
        return boxes

    def create_detected_object(self, img: np.array, detected_by: str, box: DetectionBox) -> BaseDetectedObject:
        ret = Coco91DetectedObject(img, box.confidence, box.cls_idx - 1)  # self.net.GetClassDesc(box.cls_idx)
        return ret

    def get_detected_object_class_name(self, cls_idx: int) -> str:
        return self.net.GetClassDesc(max(cls_idx - 1, 0))
