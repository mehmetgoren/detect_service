from typing import List
from PIL import Image

from pycoral.adapters import common
from pycoral.adapters import detect

from core.models.detections import DetectionResult, DetectionBox, BaseDetector


class CoralObjectDetector(BaseDetector):
    def __init__(self, interpreter, labels):
        self.interpreter = interpreter
        self.labels = labels

    # noinspection DuplicatedCode
    def detect(self, img: any) -> List[DetectionResult]:
        image = Image.fromarray(img, 'RGB')
        # Convert image to expected format
        _, scale = common.set_resized_input(
            self.interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS)
        )
        
        # Run inference
        self.interpreter.invoke()
        
        # Get detection results
        results = detect.get_objects(self.interpreter, score_threshold=0.5, image_scale=scale)
        
        # Transform results into DetectionResult objects
        ret: List[DetectionResult] = []
        for r in results:
            box = DetectionBox()
            box.x1, box.y1, box.x2, box.y2 = r.bbox
            result = DetectionResult()
            result.box = box
            result.pred_cls_name, result.pred_cls_idx, result.pred_score = (
                self.labels[r.id], r.id, r.score
            )
            ret.append(result)
        
        return ret
