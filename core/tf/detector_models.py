from abc import ABC, abstractmethod
from typing import List

import numpy as np
import tensorflow as tf

from common.utilities import logger
from core.models.coco_objects import coco91_object
from core.models.detections import DetectionResult, DetectionBox

tf_lite_models = {
    'efficientdet/lite0/detection': 'https://tfhub.dev/tensorflow/efficientdet/lite0/detection/1',
    'efficientdet/lite1/detection': 'https://tfhub.dev/tensorflow/efficientdet/lite1/detection/1',
    'efficientdet/lite2/detection': 'https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1',
    'efficientdet/lite3/detection': 'https://tfhub.dev/tensorflow/efficientdet/lite3/detection/1',
    'efficientdet/lite3x/detection': 'https://tfhub.dev/tensorflow/efficientdet/lite3x/detection/1',
    'efficientdet/lite4/detection': 'https://tfhub.dev/tensorflow/efficientdet/lite4/detection/2'  # best one
}

tf_full_models = {
    'CenterNet HourGlass104 512x512': 'https://tfhub.dev/tensorflow/centernet/hourglass_512x512/1',  # good
    'CenterNet HourGlass104 Keypoints 512x512': 'https://tfhub.dev/tensorflow/centernet/hourglass_512x512_kpts/1',
    'CenterNet HourGlass104 1024x1024': 'https://tfhub.dev/tensorflow/centernet/hourglass_1024x1024/1',
    'CenterNet HourGlass104 Keypoints 1024x1024': 'https://tfhub.dev/tensorflow/centernet/hourglass_1024x1024_kpts/1',
    'CenterNet Resnet50 V1 FPN 512x512': 'https://tfhub.dev/tensorflow/centernet/resnet50v1_fpn_512x512/1',
    'CenterNet Resnet50 V1 FPN Keypoints 512x512': 'https://tfhub.dev/tensorflow/centernet/resnet50v1_fpn_512x512_kpts/1',
    'CenterNet Resnet101 V1 FPN 512x512': 'https://tfhub.dev/tensorflow/centernet/resnet101v1_fpn_512x512/1',  # good
    'CenterNet Resnet50 V2 512x512': 'https://tfhub.dev/tensorflow/centernet/resnet50v2_512x512/1',
    'CenterNet Resnet50 V2 Keypoints 512x512': 'https://tfhub.dev/tensorflow/centernet/resnet50v2_512x512_kpts/1',
    'EfficientDet D0 512x512': 'https://tfhub.dev/tensorflow/efficientdet/d0/1',
    'EfficientDet D1 640x640': 'https://tfhub.dev/tensorflow/efficientdet/d1/1',  # good
    'EfficientDet D2 768x768': 'https://tfhub.dev/tensorflow/efficientdet/d2/1',  # good
    'EfficientDet D3 896x896': 'https://tfhub.dev/tensorflow/efficientdet/d3/1',  # good
    'EfficientDet D4 1024x1024': 'https://tfhub.dev/tensorflow/efficientdet/d4/1',  # best
    'EfficientDet D5 1280x1280': 'https://tfhub.dev/tensorflow/efficientdet/d5/1',  # best
    'EfficientDet D6 1280x1280': 'https://tfhub.dev/tensorflow/efficientdet/d6/1',  # best
    'EfficientDet D7 1536x1536': 'https://tfhub.dev/tensorflow/efficientdet/d7/1',  # best of best
    'SSD MobileNet v2 320x320': 'https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2',  # good
    'SSD MobileNet V1 FPN 640x640': 'https://tfhub.dev/tensorflow/ssd_mobilenet_v1/fpn_640x640/1',
    'SSD MobileNet V2 FPNLite 320x320': 'https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1',
    'SSD MobileNet V2 FPNLite 640x640': 'https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_640x640/1',
    'SSD ResNet50 V1 FPN 640x640 (RetinaNet50)': 'https://tfhub.dev/tensorflow/retinanet/resnet50_v1_fpn_640x640/1',
    'SSD ResNet50 V1 FPN 1024x1024 (RetinaNet50)': 'https://tfhub.dev/tensorflow/retinanet/resnet50_v1_fpn_1024x1024/1',  # best
    'SSD ResNet101 V1 FPN 640x640 (RetinaNet101)': 'https://tfhub.dev/tensorflow/retinanet/resnet101_v1_fpn_640x640/1',
    'SSD ResNet101 V1 FPN 1024x1024 (RetinaNet101)': 'https://tfhub.dev/tensorflow/retinanet/resnet101_v1_fpn_1024x1024/1',  # good
    'SSD ResNet152 V1 FPN 640x640 (RetinaNet152)': 'https://tfhub.dev/tensorflow/retinanet/resnet152_v1_fpn_640x640/1',
    'SSD ResNet152 V1 FPN 1024x1024 (RetinaNet152)': 'https://tfhub.dev/tensorflow/retinanet/resnet152_v1_fpn_1024x1024/1',  # good
    'Faster R-CNN ResNet50 V1 640x640': 'https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1',  # good
    'Faster R-CNN ResNet50 V1 1024x1024': 'https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_1024x1024/1',
    'Faster R-CNN ResNet50 V1 800x1333': 'https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_800x1333/1',  # very good
    'Faster R-CNN ResNet101 V1 640x640': 'https://tfhub.dev/tensorflow/faster_rcnn/resnet101_v1_640x640/1',
    'Faster R-CNN ResNet101 V1 1024x1024': 'https://tfhub.dev/tensorflow/faster_rcnn/resnet101_v1_1024x1024/1',  # good
    'Faster R-CNN ResNet101 V1 800x1333': 'https://tfhub.dev/tensorflow/faster_rcnn/resnet101_v1_800x1333/1',  # good
    'Faster R-CNN ResNet152 V1 640x640': 'https://tfhub.dev/tensorflow/faster_rcnn/resnet152_v1_640x640/1',  # good
    'Faster R-CNN ResNet152 V1 1024x1024': 'https://tfhub.dev/tensorflow/faster_rcnn/resnet152_v1_1024x1024/1',  # good
    'Faster R-CNN ResNet152 V1 800x1333': 'https://tfhub.dev/tensorflow/faster_rcnn/resnet152_v1_800x1333/1',  # good
    'Faster R-CNN Inception ResNet V2 640x640': 'https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_640x640/1',  # good
    'Faster R-CNN Inception ResNet V2 1024x1024': 'https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_1024x1024/1',  # good
    'Mask R-CNN Inception ResNet V2 1024x1024': 'https://tfhub.dev/tensorflow/mask_rcnn/inception_resnet_v2_1024x1024/1'  # good
}


class BoxPoints:
    def __init__(self, x1: int, y1: int, x2: int, y2: int):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2


class BaseTfModel(ABC):
    def __init__(self, hub_model):
        self.hub_model = hub_model

    def detect(self, img_np: np.array) -> List[DetectionResult]:
        ret: List[DetectionResult] = []
        try:
            # Convert img to RGB
            rgb_tensor = tf.convert_to_tensor(img_np, dtype=tf.uint8)
            # Add dims to rgb_tensor
            rgb_tensor = tf.expand_dims(rgb_tensor, 0)

            detections = self.hub_model(rgb_tensor)
            boxes, scores, classes = self._parse_detections(detections)
            for j, score in enumerate(scores):
                cls_idx = int(classes[j]) - 1
                cls_name = coco91_object.get_name(cls_idx)

                box = boxes[j]
                points = self._parse_points(img_np, box)
                b = DetectionBox()
                b.x1, b.y1, b.x2, b.y2 = int(points.x1), int(points.y1), int(points.x2), int(points.y2)
                r = DetectionResult()
                r.box = b
                r.pred_cls_name, r.pred_cls_idx, r.pred_score = cls_name, cls_idx, float(score)
                ret.append(r)
        except BaseException as ex:
            logger.error(f'an error occurred while tensorflow detection call, ex: {ex}')
        return ret

    @abstractmethod
    def _parse_detections(self, detections):
        raise NotImplementedError('BaseTfModel._parse_detections')

    @abstractmethod
    def _parse_points(self, original_np_img: np.array, detections) -> BoxPoints:
        raise NotImplementedError('BaseTfModel._parse_boxes')


class TfModelLite(BaseTfModel):
    def __init__(self, hub_model):
        super().__init__(hub_model)
        self.model_name: str = ''

    def _parse_detections(self, detections):
        boxes, scores, classes, num_detections = detections
        return boxes.numpy()[0], scores.numpy()[0], classes.numpy()[0]

    def _parse_points(self, original_np_img: np.array, box) -> BoxPoints:
        y1, x1, y2, x2 = box
        box_points = BoxPoints(int(x1), int(y1), int(x2), int(y2))
        return box_points


class TfModelFull(BaseTfModel):
    def __init__(self, hub_model):
        super().__init__(hub_model)
        self.model_name: str = ''

    def _parse_detections(self, detections):
        boxes, scores, classes = detections['detection_boxes'], detections['detection_scores'], detections['detection_classes']
        return boxes.numpy()[0], scores.numpy()[0], classes.numpy()[0]

    def _parse_points(self, original_np_img: np.array, box) -> BoxPoints:
        height, width = original_np_img.shape[0], original_np_img.shape[1]
        y1, x1, y2, x2 = box
        x1, x2, y1, y2 = int(x1 * width), int(x2 * width), int(y1 * height), int(y2 * height)
        box_points = BoxPoints(x1, y1, x2, y2)
        return box_points
