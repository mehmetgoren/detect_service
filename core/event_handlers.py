import base64
import json
import cv2
from threading import Thread
import numpy as np
from typing import List

from common.event_bus.event_bus import EventBus
from common.event_bus.event_handler import EventHandler
from common.utilities import logger, config
from core.models.detected_objects import BaseDetectedObject
from core.object_detectors.object_detector import ObjectDetector
from core.object_framers import ObjectFramerBase


class SaveImageEventHandler(EventHandler):
    def __init__(self):
        self.folder_path = config.handler.save_image_folder_path
        self.file_extension = config.handler.save_image_extension

    def handle(self, detected: BaseDetectedObject):
        key = detected.create_unique_key()
        file_name = f'{self.folder_path}/{key}.{self.file_extension}'
        cv2.imwrite(file_name, detected.get_image())


class ShowImageEventHandler(EventHandler):
    def __init__(self):
        self.wait_key = config.handler.show_image_wait_key
        self.caption = config.handler.show_image_caption
        self.fullscreen = config.handler.show_image_fullscreen

    def handle(self, detected: BaseDetectedObject):
        caption = detected.get_text() if self.caption else 'window'
        if self.fullscreen:
            cv2.namedWindow(caption, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(caption, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(caption, detected.get_image())
        return cv2.waitKey(self.wait_key)


class ReadServiceEventHandler(EventHandler):
    def __init__(self, detector: ObjectDetector, framer: ObjectFramerBase):
        self.detector = detector
        self.framer = framer
        self.channel = 'read'
        self.redis_event_handler = RedisEventHandler()
        self.encoding = 'utf-8'
        self.overlay = config.handler.read_service_overlay

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
        img_str = dic['img']
        jpg_original = base64.b64decode(img_str)
        jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
        img = cv2.imdecode(jpg_as_np, flags=1)

        detected_list: List[BaseDetectedObject] = self.framer.frame(self.detector, img, name)
        if len(detected_list):
            for detected in detected_list:
                detected.detected_by = name

                dic = {'file_name': detected.create_unique_key()}

                if self.overlay:
                    img_to_bytes = cv2.imencode('.jpg', detected.get_image())
                    if not len(img_to_bytes) != 1:
                        logger.warning(f'img_to_bytes length is insufficient: {len(img_to_bytes)}')
                        return
                    img_to_bytes = img_to_bytes[1].tobytes()
                    base64_img = base64.b64encode(img_to_bytes)
                    dic['base64_image'] = base64_img.decode(self.encoding)
                else:
                    dic['base64_image'] = img_str

                self.redis_event_handler.handle(dic)
        else:
            logger.info(f'(camera {name}) detected nothing')


# it's kinda proxy for EventBus
class RedisEventHandler(EventHandler):
    def __init__(self):
        self.channel = 'detect'
        self.event_bus = EventBus(self.channel)

    def handle(self, dic: dict):
        event = json.dumps(dic)
        self.event_bus.publish(event)

    # def _handle(self, detected: BaseDetectedObject):
    #     key = detected.create_unique_key()
    #     img_to_bytes = cv2.imencode('.jpg', detected.get_image())
    #     if not len(img_to_bytes) != 1:
    #         logger.warning(f'img_to_bytes length is insufficient: {len(img_to_bytes)}')
    #         return
    #     img_to_bytes = img_to_bytes[1].tobytes()
    #     base64_img = base64.b64encode(img_to_bytes)
    #     dic = {'file_name': key, 'base64_image': base64_img.decode(self.encoding)}
    #     event = json.dumps(dic)
    #     self.event_bus.publish(event)
