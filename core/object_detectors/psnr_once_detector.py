import numpy as np
import cv2
from redis.client import Redis

from common.utilities import config
from core.object_detectors.base_once_detector import BaseOnceDetector
from core.models.object_detector_model import BaseObjectDetectorModel


class PsnrOnceDetector(BaseOnceDetector):
    def __init__(self, connection: Redis, detector_model: BaseObjectDetectorModel):
        super(PsnrOnceDetector, self).__init__(connection, detector_model)
        self.psnr_threshold = config.once_detector.psnr_threshold

    def _process_img(self, whole_img: np.array):
        return whole_img

    def _get_loss(self, processed_img: np.array, prev_processed_img: np.array):
        psnr = cv2.PSNR(processed_img, prev_processed_img)
        loss = 20.0 - psnr
        return loss

    def _get_algorithm_threshold(self):
        return self.psnr_threshold
