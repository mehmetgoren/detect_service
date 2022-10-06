from abc import ABC, abstractmethod

from core.models.coco_objects import coco80_object, coco91_object


class BaseOdObject(ABC):
    def __init__(self, pred_score: float, pred_cls_idx: int):
        self.pred_score = pred_score
        self.pred_cls_idx = pred_cls_idx
        self.detected_by = None
        self.separator = '_'

    def get_text(self) -> str:
        text = f'{self.get_pred_cls_name()}{self.separator} {"{:.2f}".format(self.pred_score)}'
        return text

    def get_pred_score(self) -> float:
        return self.pred_score

    def get_pred_cls_index(self) -> int:
        return self.pred_cls_idx

    @abstractmethod
    def _get_pred_cls_name(self, pred_cls_idx: int) -> str:
        raise NotImplementedError('BaseOdObject.get_pred_cls_name()')

    def get_pred_cls_name(self) -> str:
        if self.pred_cls_idx is None:
            return ''
        return self._get_pred_cls_name(self.pred_cls_idx)


class Coco80OdObject(BaseOdObject):
    def __init__(self, pred_score: float, pred_cls_idx: int):
        super(Coco80OdObject, self).__init__(pred_score, pred_cls_idx)

    def _get_pred_cls_name(self, pred_cls_idx: int) -> str:
        return coco80_object.get_name(pred_cls_idx)


class Coco91OdObject(BaseOdObject):
    def __init__(self, pred_score: float, pred_cls_idx: int):
        super(Coco91OdObject, self).__init__(pred_score, pred_cls_idx)

    def _get_pred_cls_name(self, pred_cls_idx: int) -> str:
        return coco91_object.get_name(pred_cls_idx)
