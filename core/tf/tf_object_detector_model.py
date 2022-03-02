import os
from typing import List
import numpy as np
import tensorflow_hub as hub

from common.utilities import config
from core.models.detected_objects import Coco80DetectedObject, coco80_info, BaseDetectedObject
from core.models.object_detector_model import BaseObjectDetectorModel, DetectionBox
from core.tf.detector_models import tf_lite_models, tf_full_models, TfModelLite, TfModelFull


class TfObjectDetectorModel(BaseObjectDetectorModel):
    def __init__(self):
        super(TfObjectDetectorModel, self).__init__()
        os.environ['TFHUB_CACHE_DIR'] = config.tensorflow.cache_folder
        self.model_name = config.tensorflow.model_name
        self.is_lite = self.model_name in tf_lite_models
        self.hub_model = hub.load(tf_lite_models[self.model_name] if self.is_lite else tf_full_models[self.model_name])
        self.threshold = config.tensorflow.threshold
        self.model = TfModelLite(self.hub_model, self.threshold) if self.is_lite else TfModelFull(self.hub_model, self.threshold)

    def get_detect_boxes(self, img: np.array, detected_by: str) -> List[DetectionBox]:
        return self.model.detect(img)

    def create_detected_object(self, img: np.array, detected_by: str, box: DetectionBox) -> BaseDetectedObject:
        obj = Coco80DetectedObject(img, box.confidence, box.cls_idx - 1)
        obj.detected_by = detected_by
        return obj

    def get_detected_object_class_name(self, cls_idx: int) -> str:
        return coco80_info.get_name(cls_idx - 1)
