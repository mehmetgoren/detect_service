import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

from common.config import DeviceType
from common.utilities import config
from core.object_detectors.base_once_detector import BaseOnceDetector
from core.models.object_detector_model import BaseObjectDetectorModel


# info: need to use SSIM technique to accept different people in the same box area.
# info: Also make sure it performant good on jetson nano.
# info: gray images compression takes 31ms(multi-channel is 90ms) on ryzen 3000 cpu time
class SsimOnceDetector(BaseOnceDetector):
    def __init__(self, device: DeviceType, detector_model: BaseObjectDetectorModel):
        super(SsimOnceDetector, self).__init__(device, detector_model)
        self.ssim_threshold = config.once_detector.ssim_threshold

    def _process_img(self, whole_img: np.array):  # make it gray
        return cv2.cvtColor(whole_img, cv2.COLOR_BGR2GRAY)

    def _get_loss(self, processed_img: np.array, prev_processed_img: np.array):
        loss = ssim(processed_img, prev_processed_img)
        loss = 1.0 - loss
        return loss

    def _get_algorithm_threshold(self):
        return self.ssim_threshold
