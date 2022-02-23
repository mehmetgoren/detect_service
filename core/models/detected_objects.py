from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, List
import matplotlib
from PIL import ImageColor
from datetime import datetime
import uuid


class BaseCocoInfo(ABC):
    def __init__(self):
        self.names = None
        self.colors = None

    @abstractmethod
    def _create_coco_names(self) -> List[str]:
        raise NotImplementedError('BaseCocoInfo._create_coco_names')

    def get_names(self) -> List[str]:
        if self.names is None:
            self.names = self._create_coco_names()
        return self.names

    def get_name(self, index: int) -> str:
        return self.get_names()[index]

    def get_colors(self) -> List[Tuple[int, int, int]]:
        if self.colors is None:
            hex_list = list(matplotlib.colors.cnames.values())
            self.colors = [ImageColor.getrgb(hex) for hex in hex_list]
        return self.colors

    def get_color(self, index: int) -> Tuple[int, int, int]:
        return self.get_colors()[index]


class Coco80Info(BaseCocoInfo):
    def __init__(self):
        super().__init__()

    def _create_coco_names(self) -> List[str]:
        return ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa',
                'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard',
                'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                'teddy bear', 'hair drier', 'toothbrush']


coco80_info = Coco80Info()


class Coco91Info(BaseCocoInfo):
    def __init__(self):
        super().__init__()

    def _create_coco_names(self) -> List[str]:
        return ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                'traffic light', 'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench', 'bird',
                'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat',
                'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife',
                'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
                'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table',
                'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book',
                'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'hair brush']


coco91_info = Coco91Info()


class BaseDetectedObject(ABC):
    @abstractmethod
    def get_image(self) -> np.array:
        raise NotImplementedError('BaseDetectedObject.get_text()')

    @abstractmethod
    def get_text(self) -> str:
        raise NotImplementedError('BaseDetectedObject.get_text()')

    @abstractmethod
    def create_unique_key(self) -> str:
        raise NotImplementedError('BaseDetectedObject.create_unique_key()')

    @abstractmethod
    def get_pred_cls_name(self) -> str:
        raise NotImplementedError('BaseDetectedObject.get_pred_cls_name()')

    @abstractmethod
    def get_pred_color(self) -> Tuple[int, int, int]:
        raise NotImplementedError('BaseDetectedObject.get_pred_color()')


class BaseCocoDetectedObject(BaseDetectedObject):
    def __init__(self, img: np.array, pred_score: float, pred_cls_idx: int):
        super(BaseCocoDetectedObject, self).__init__()
        self.img: np.array = img
        self.pred_score = pred_score
        self.pred_cls_idx = pred_cls_idx
        self.track_id = None
        self.detected_by = None

    def get_image(self) -> np.array:
        return self.img

    def get_text(self) -> str:
        text = self.get_pred_cls_name() + (
            '_' + str(self.track_id) if self.track_id is not None else '') + ' ' + "{:.2f}".format(self.pred_score)
        return text

    def create_unique_key(self) -> str:
        strings = [''] * 6
        strings[0] = (str(self.detected_by) + '_' if self.detected_by is not None else '')
        strings[1] = self.get_pred_cls_name() + '_'
        strings[2] = (str(self.track_id) + '_' if self.track_id is not None else '')
        strings[3] = '{:.2f}'.format(self.pred_score) + '_'
        strings[4] = datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3] + '_'
        strings[5] = str(uuid.uuid4().hex)
        return ''.join(strings)

    @abstractmethod
    def _get_pred_cls_name(self, pred_cls_idx: int) -> str:
        raise NotImplementedError('BaseCocoDetectedObject.get_pred_cls_name()')

    def get_pred_cls_name(self) -> str:
        if self.pred_cls_idx is None:
            return ''
        return self._get_pred_cls_name(self.pred_cls_idx)

    @abstractmethod
    def _get_pred_color(self, pred_cls_idx: int) -> Tuple[int, int, int]:
        raise NotImplementedError('BaseCocoDetectedObject.get_pred_color()')

    def get_pred_color(self) -> Tuple[int, int, int]:
        if self.pred_cls_idx < 0:
            return 0, 0, 0
        return self._get_pred_color(self.pred_cls_idx)


class Coco80DetectedObject(BaseCocoDetectedObject):
    def __init__(self, img: np.array, pred_score: float, pred_cls_idx: int):
        super(Coco80DetectedObject, self).__init__(img, pred_score, pred_cls_idx)

    def _get_pred_cls_name(self, pred_cls_idx: int) -> str:
        return coco80_info.get_name(pred_cls_idx)

    def _get_pred_color(self, pred_cls_idx: int) -> Tuple[int, int, int]:
        return coco80_info.get_color(pred_cls_idx)


class Coco91DetectedObject(BaseCocoDetectedObject):
    def __init__(self, img: np.array, pred_score: float, pred_cls_idx: int):
        super(Coco91DetectedObject, self).__init__(img, pred_score, pred_cls_idx)

    def _get_pred_cls_name(self, pred_cls_idx: int) -> str:
        return coco91_info.get_name(pred_cls_idx)

    def _get_pred_color(self, pred_cls_idx: int) -> Tuple[int, int, int]:
        return coco91_info.get_color(pred_cls_idx)
