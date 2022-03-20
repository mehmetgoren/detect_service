import base64
import io
import json
from enum import IntEnum
from threading import Thread
import numpy as np
from typing import List
from PIL import Image, UnidentifiedImageError
from redis.client import Redis

from common.event_bus.event_bus import EventBus
from common.event_bus.event_handler import EventHandler
from common.utilities import logger, config
from core.data.od_cache import OdCache
from core.models.detected_objects import BaseDetectedObject
from core.object_detectors.object_detector import ObjectDetector
from core.object_framers import ObjectFramerBase


class ModelChanged:
    def __init__(self):
        self.source_id: str = ''


class ModelChangedOp(IntEnum):
    SAVE = 0
    DELETE = 1


class DataChangedEvent:
    def __init__(self):
        self.model_name: str = ''
        self.params_json: str = ''
        self.op: ModelChangedOp = ModelChangedOp.SAVE


class DataChangedEventHandler(EventHandler):
    def __init__(self, connection: Redis):
        self.channel = 'data_changed'
        self.encoding = 'utf-8'
        self.cache = OdCache(connection)

    def handle(self, dic: dict):
        if dic is None or dic['type'] != 'message':
            return

        data: bytes = dic['data']
        dic = json.loads(data.decode(self.encoding))
        event = DataChangedEvent()
        event.__dict__.update(dic)
        if event.model_name != 'od':
            return
        mc = ModelChanged()
        dic = json.loads(event.params_json)
        mc.__dict__.update(dic)
        if event.op == ModelChangedOp.SAVE:
            self.cache.refresh(mc.source_id)
        elif event.op == ModelChangedOp.DELETE:
            self.cache.remove(mc.source_id)
        else:
            raise NotImplementedError(event.op)


class ReadServiceEventHandler(EventHandler):
    def __init__(self, detector: ObjectDetector, framer: ObjectFramerBase):
        self.detector = detector
        self.framer = framer
        self.encoding = 'utf-8'
        self.overlay = config.ai.read_service_overlay
        self.publisher = EventBus('detect_service')

    def handle(self, dic: dict):
        if dic is None or dic['type'] != 'message':
            return

        th = Thread(target=self._handle, args=[dic])
        th.daemon = True
        th.start()

    def _handle(self, dic: dict):
        data: bytes = dic['data']
        dic = json.loads(data.decode(self.encoding))
        name = dic['name']
        source_id = dic['source']
        img_str = dic['img']
        base64_decoded = base64.b64decode(img_str)
        try:
            image = Image.open(io.BytesIO(base64_decoded))
        except UnidentifiedImageError as err:
            logger.error(f'an error occurred while creating a PIL image from base64 string, err: {err}')
            return
        img_np = np.asarray(image)

        detected_list: List[BaseDetectedObject] = self.framer.frame(self.detector, img_np, source_id)
        if len(detected_list):
            for detected in detected_list:
                detected.detected_by = source_id
                dic = {'file_name': detected.create_unique_key()}

                if self.overlay:
                    img = Image.fromarray(img_np)
                    buffered = io.BytesIO()
                    img.save(buffered, format="JPEG")
                    img_to_bytes = buffered.getvalue()
                    if not len(img_to_bytes) != 1:
                        logger.warning(f'img_to_bytes length is insufficient: {len(img_to_bytes)}')
                        return
                    dic['base64_image'] = base64.b64encode(img_to_bytes).decode()
                else:
                    dic['base64_image'] = img_str

                event = json.dumps(dic)
                self.publisher.publish(event)
        else:
            logger.info(f'(camera {name}) detected nothing')
