from abc import ABC, abstractmethod
from typing import List


class BaseCocoObject(ABC):
    def __init__(self):
        self.names = None
        self.indexes = None

    @abstractmethod
    def _create_coco_names(self) -> List[str]:
        raise NotImplementedError('BaseCocoInfo._create_coco_names')

    def _create_coco_indexes(self) -> dict:
        ret = dict()
        names = self._create_coco_names()
        for index, name in enumerate(names):
            ret[name] = index
        return ret

    def get_name(self, cls_idx: int) -> str:
        if self.names is None:
            self.names = self._create_coco_names()
        return self.names[cls_idx]

    def get_index(self, name: str) -> int:
        if self.indexes is None:
            self.indexes = self._create_coco_indexes()
        return self.indexes[name]


class Coco80Object(BaseCocoObject):
    def __init__(self):
        super().__init__()

    def _create_coco_names(self) -> List[str]:
        return ['person', 'bicycle', 'car', 'motorcycle', 'airplane',
                'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop_sign',
                'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
                'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
                'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
                'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donot', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
                'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                'hair dryer', 'toothbrush']


coco80_object = Coco80Object()


class Coco91Object(BaseCocoObject):
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


coco91_object = Coco91Object()
