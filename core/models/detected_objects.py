from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, List
from PIL import ImageColor


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

    @staticmethod
    def _create_colors() -> List[str]:
        return ['#F0F8FF', '#FAEBD7', '#00FFFF', '#7FFFD4', '#F0FFFF', '#F5F5DC', '#FFE4C4', '#000000', '#FFEBCD', '#0000FF', '#8A2BE2', '#A52A2A', '#DEB887',
                '#5F9EA0', '#7FFF00', '#D2691E', '#FF7F50', '#6495ED', '#FFF8DC', '#DC143C', '#00FFFF', '#00008B', '#008B8B', '#B8860B', '#A9A9A9', '#006400',
                '#A9A9A9', '#BDB76B', '#8B008B', '#556B2F', '#FF8C00', '#9932CC', '#8B0000', '#E9967A', '#8FBC8F', '#483D8B', '#2F4F4F', '#2F4F4F', '#00CED1',
                '#9400D3', '#FF1493', '#00BFFF', '#696969', '#696969', '#1E90FF', '#B22222', '#FFFAF0', '#228B22', '#FF00FF', '#DCDCDC', '#F8F8FF', '#FFD700',
                '#DAA520', '#808080', '#008000', '#ADFF2F', '#808080', '#F0FFF0', '#FF69B4', '#CD5C5C', '#4B0082', '#FFFFF0', '#F0E68C', '#E6E6FA', '#FFF0F5',
                '#7CFC00', '#FFFACD', '#ADD8E6', '#F08080', '#E0FFFF', '#FAFAD2', '#D3D3D3', '#90EE90', '#D3D3D3', '#FFB6C1', '#FFA07A', '#20B2AA', '#87CEFA',
                '#778899', '#778899', '#B0C4DE', '#FFFFE0', '#00FF00', '#32CD32', '#FAF0E6', '#FF00FF', '#800000', '#66CDAA', '#0000CD', '#BA55D3', '#9370DB',
                '#3CB371', '#7B68EE', '#00FA9A', '#48D1CC', '#C71585', '#191970', '#F5FFFA', '#FFE4E1', '#FFE4B5', '#FFDEAD', '#000080', '#FDF5E6', '#808000',
                '#6B8E23', '#FFA500', '#FF4500', '#DA70D6', '#EEE8AA', '#98FB98', '#AFEEEE', '#DB7093', '#FFEFD5', '#FFDAB9', '#CD853F', '#FFC0CB', '#DDA0DD',
                '#B0E0E6', '#800080', '#663399', '#FF0000', '#BC8F8F', '#4169E1', '#8B4513', '#FA8072', '#F4A460', '#2E8B57', '#FFF5EE', '#A0522D', '#C0C0C0',
                '#87CEEB', '#6A5ACD', '#708090', '#708090', '#FFFAFA', '#00FF7F', '#4682B4', '#D2B48C', '#008080', '#D8BFD8', '#FF6347', '#40E0D0', '#EE82EE',
                '#F5DEB3', '#FFFFFF', '#F5F5F5', '#FFFF00', '#9ACD32']

    def get_colors(self) -> List[Tuple[int, int, int]]:
        if self.colors is None:
            hex_list = self._create_colors()
            self.colors = [ImageColor.getrgb(hx) for hx in hex_list]
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
    def get_pred_score(self) -> float:
        raise NotImplementedError('BaseDetectedObject.get_pred_score()')

    @abstractmethod
    def get_pred_cls_index(self) -> int:
        raise NotImplementedError('BaseDetectedObject.get_pred_cls_index()')

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
        self.separator = '_'

    def get_image(self) -> np.array:
        return self.img

    def get_text(self) -> str:
        text = self.get_pred_cls_name() + (
            self.separator + str(self.track_id) if self.track_id is not None else '') + ' ' + "{:.2f}".format(self.pred_score)
        return text

    def get_pred_score(self) -> float:
        return self.pred_score

    def get_pred_cls_index(self) -> int:
        return self.pred_cls_idx

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
