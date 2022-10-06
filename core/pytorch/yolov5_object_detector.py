from typing import List
import numpy.typing as npt

from common.utilities import logger
from core.models.coco_objects import coco80_object
from core.models.detections import DetectionBox, DetectionResult, BaseDetector


class Yolov5ObjectDetector(BaseDetector):
    def __init__(self, model):
        self.model = model

    # noinspection DuplicatedCode
    def detect(self, img: npt.NDArray) -> List[DetectionResult]:
        detections = self.model(img)
        # print img1 predictions (pixels)
        #                   x1           y1           x2           y2   confidence        class
        # tensor([[7.50637e+02, 4.37279e+01, 1.15887e+03, 7.08682e+02, 8.18137e-01, 0.00000e+00],
        #         [9.33597e+01, 2.07387e+02, 1.04737e+03, 7.10224e+02, 5.78011e-01, 0.00000e+00],
        #         [4.24503e+02, 4.29092e+02, 5.16300e+02, 7.16425e+02, 5.68713e-01, 2.70000e+01]])
        # Output will be a numpy array in the following format:
        # [[x1, y1, x2, y2, confidence, class]]
        ret: List[DetectionResult] = []
        try:
            for box in detections.xyxy[0]:
                x1, y1, x2, y2, conf, cls_idx = int(box[0].item()), int(box[1].item()), int(box[2].item()), int(box[3].item()), box[4].item(), int(
                    box[5].item())
                cls_name = coco80_object.get_name(cls_idx)
                box = DetectionBox()
                box.x1, box.y1, box.x2, box.y2 = x1, y1, x2, y2
                r = DetectionResult()
                r.box = box
                r.pred_cls_name, r.pred_cls_idx, r.pred_score = cls_name, cls_idx, conf
                ret.append(r)
        except BaseException as ex:
            logger.error(f'an error occurred while detection api call, ex: {ex}')
        return ret
