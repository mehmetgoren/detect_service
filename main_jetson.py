from core.event_handlers import ReadServiceEventHandler
from core.jetson.jetson_detector import JetsonObjectDetector
from core.jetson.model_loader import load_model
from common.event_bus.event_bus import EventBus
from common.utilities import logger
from core.models.detections import BaseDetector
from core.utilities import register_detect_service, EventChannels


def create_object_detector_model() -> BaseDetector:
    model = load_model()
    return JetsonObjectDetector(model)


def main():
    register_detect_service('jetson_detection_service', 'detect_service_jetson-instance', 'The Jetson Object Detection Service®')

    detector = create_object_detector_model()
    handler = ReadServiceEventHandler(detector)

    logger.info('jetson object detection service will start soon')
    event_bus = EventBus(EventChannels.snapshot_in)
    event_bus.subscribe_async(handler)


if __name__ == '__main__':
    main()
