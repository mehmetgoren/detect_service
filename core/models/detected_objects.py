from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, List
import matplotlib
from PIL import ImageColor
from datetime import datetime
import uuid


class Coco80Info:
    def __init__(self):
        self.names = None
        self.colors = None

    def get_names(self) -> List[str]:
        if self.names is None:
            self.names = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat',
                          'traffic light',
                          'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
                          'cow',
                          'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
                          'frisbee',
                          'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                          'surfboard',
                          'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
                          'apple',
                          'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                          'sofa',
                          'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote',
                          'keyboard',
                          'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                          'scissors',
                          'teddy bear', 'hair drier', 'toothbrush']
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


coco80_info = Coco80Info()


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
    def __init__(self, img: np.array, pred_score: float):
        super(BaseCocoDetectedObject, self).__init__()
        self.img: np.array = img
        self.pred_score = pred_score
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
    def get_pred_cls_name(self) -> str:
        raise NotImplementedError('BaseCocoDetectedObject.get_pred_cls_name()')

    @abstractmethod
    def get_pred_color(self) -> Tuple[int, int, int]:
        raise NotImplementedError('BaseCocoDetectedObject.get_pred_color()')


class Coco80DetectedObject(BaseCocoDetectedObject):
    def __init__(self, img: np.array, pred_score: float, pred_cls_idx: int):
        super(Coco80DetectedObject, self).__init__(img, pred_score)
        self.pred_cls_idx = pred_cls_idx

    def get_pred_cls_name(self) -> str:
        if self.pred_cls_idx is None:
            return ''
        return coco80_info.get_name(self.pred_cls_idx)

    def get_pred_color(self) -> Tuple[int, int, int]:
        if self.pred_cls_idx < 0:
            return 0, 0, 0
        return coco80_info.get_color(self.pred_cls_idx)


class Coco91DetectedObject(BaseCocoDetectedObject):
    def __init__(self, img: np.array, pred_score: float, pred_cls_idx: int, pred_cls_name: str):
        super(Coco91DetectedObject, self).__init__(img, pred_score)
        self.pred_cls_idx = pred_cls_idx
        self.pred_cls_name = pred_cls_name
        self.names = None

    def get_names(self) -> List[str]:
        if self.names is None:
            self.names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
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
        return self.names

    def get_pred_cls_name(self) -> str:
        return self.pred_cls_name if self.pred_cls_name is not None else self.get_names()[self.pred_cls_idx]

    def get_pred_color(self) -> Tuple[int, int, int]:
        if self.pred_cls_idx < 0:
            return 0, 0, 0
        return coco80_info.get_color(self.pred_cls_idx)
