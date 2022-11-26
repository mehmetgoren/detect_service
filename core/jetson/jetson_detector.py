from typing import List
import jetson.utils as utils

from core.models.detections import DetectionResult, DetectionBox, BaseDetector


class JetsonObjectDetector(BaseDetector):
    def __init__(self, net):
        self.net = net

    # noinspection DuplicatedCode
    def detect(self, img: any) -> List[DetectionResult]:
        img_cuda = utils.cudaFromNumpy(img)
        detections = self.net.Detect(img_cuda, overlay='OVERLAY_NONE')

        ret: List[DetectionResult] = []
        for d in detections:
            x1, y1, x2, y2 = int(d.Left), int(d.Top), int(d.Right), int(d.Bottom)
            conf, cls_idx = d.Confidence, d.ClassID
            cls_name = self.net.GetClassDesc(cls_idx)

            box = DetectionBox()
            box.x1, box.y1, box.x2, box.y2 = x1, y1, x2, y2
            r = DetectionResult()
            r.box = box
            r.pred_cls_name, r.pred_cls_idx, r.pred_score = cls_name, cls_idx, conf
            ret.append(r)

        return ret
