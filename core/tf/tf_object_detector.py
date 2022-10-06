import os
from typing import List
import numpy as np
import tensorflow_hub as hub

from common.utilities import config
from core.models.detections import DetectionResult, BaseDetector
from core.tf.detector_models import tf_lite_models, tf_full_models, TfModelLite, TfModelFull


class TfObjectDetector(BaseDetector):
    def __init__(self):
        os.environ['TFHUB_CACHE_DIR'] = config.tensorflow.cache_folder
        self.model_name = config.tensorflow.model_name
        self.is_lite = self.model_name in tf_lite_models
        self.hub_model = hub.load(tf_lite_models[self.model_name] if self.is_lite else tf_full_models[self.model_name])
        self.model = TfModelLite(self.hub_model) if self.is_lite else TfModelFull(self.hub_model)

    def detect(self, img: np.array) -> List[DetectionResult]:
        return self.model.detect(img)
