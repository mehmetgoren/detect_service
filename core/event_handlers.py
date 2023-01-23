import base64
import io
import json
from threading import Thread
import numpy as np
from typing import List
from PIL import Image, UnidentifiedImageError

from common.event_bus.event_bus import EventBus
from common.event_bus.event_handler import EventHandler
from common.utilities import logger, config
from core.models.detections import BaseDetector, DetectionResult
from core.utilities import EventChannels


class ReadServiceEventHandler(EventHandler):
    def __init__(self, detector: BaseDetector):
        self.detector = detector
        self.encoding = 'utf-8'
        self.publisher = EventBus(EventChannels.snapshot_out)

    def handle(self, dic: dict):
        if dic is None or dic['type'] != 'message':
            return

        th = Thread(target=self._handle, args=[dic])
        th.daemon = True
        th.start()

    # you can handle it and add dict to multiprocessing.Queue then execute on a parameterless function by Pool.run_async to provide a real multi-core support
    def _handle(self, dic: dict):
        data: bytes = dic['data']
        dic = json.loads(data.decode(self.encoding))
        name = dic['name']
        source_id = dic['source_id']
        base64_image = dic['base64_image']
        ai_clip_enabled = dic['ai_clip_enabled']

        base64_decoded = base64.b64decode(base64_image)
        try:
            image = Image.open(io.BytesIO(base64_decoded))
        except UnidentifiedImageError as err:
            logger.error(f'an error occurred while creating a PIL image from base64 string, err: {err}')
            return
        img_np = np.asarray(image)

        results: List[DetectionResult] = self.detector.detect(img_np)
        if len(results) > 0:
            detected_dic_list = []
            for r in results:
                dic_box = {'x1': r.box.x1, 'y1': r.box.y1, 'x2': r.box.x2, 'y2': r.box.y2}
                dic_result = {'pred_cls_name': r.pred_cls_name, 'pred_cls_idx': r.pred_cls_idx, 'pred_score': r.pred_score, 'box': dic_box}
                detected_dic_list.append(dic_result)

            dic = {'name': name, 'source': source_id, 'img': base64_image, 'ai_clip_enabled': ai_clip_enabled, 'detections': detected_dic_list,
                   'channel': EventChannels.od_service, 'list_name': 'detected_objects'}
            event = json.dumps(dic)
            self.publisher.publish(event)
        else:
            logger.info(f'(camera {name}) detected nothing')
