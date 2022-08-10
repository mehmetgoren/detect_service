from core.object_framers import DrawObjectFramer
from core.models.object_detector_model import BaseObjectDetectorModel
from core.object_detectors.imagehash_once_detector import ImageHashOnceDetector
from core.event_handlers import ReadServiceEventHandler
from core.jetson.model_loader import load_model
from core.jetson.jetson_detector_model import JetsonbjectDetectorModel
from common.event_bus.event_bus import EventBus
from common.utilities import logger
from core.utilities import listen_data_changed_event, register_detect_service


def create_object_detector_model() -> BaseObjectDetectorModel:
    model = load_model()
    return JetsonbjectDetectorModel(model)


def main():
    conn = register_detect_service('jetson_detection_service', 'detect_service_jetson-instance', 'The Jetson Object Detection Service®')
    listen_data_changed_event(conn)

    framer = DrawObjectFramer()
    detector = ImageHashOnceDetector(conn, create_object_detector_model())
    handler = ReadServiceEventHandler(detector, framer)

    logger.info('jetson object detection service will start soon')
    event_bus = EventBus('read_service')
    event_bus.subscribe_async(handler)


if __name__ == '__main__':
    main()
