import numpy as np
import imagehash
from PIL import Image

from common.config import DeviceType
from common.utilities import config
from core.object_detectors.base_once_detector import BaseOnceDetector
from core.models.object_detector_model import BaseObjectDetectorModel


class ImageHashOnceDetector(BaseOnceDetector):
    def __init__(self, device: DeviceType, detector_model: BaseObjectDetectorModel):
        super(ImageHashOnceDetector, self).__init__(device, detector_model)
        self.imagehash_threshold = config.once_detector.imagehash_threshold

    def _process_img(self, whole_img: np.array):
        return Image.fromarray(whole_img)

    def _get_loss(self, processed_img: np.array, prev_processed_img: np.array):
        hash1 = imagehash.average_hash(processed_img)
        hash2 = imagehash.average_hash(prev_processed_img)
        loss = hash1 - hash2
        return loss

    def _get_algorithm_threshold(self):
        return self.imagehash_threshold
