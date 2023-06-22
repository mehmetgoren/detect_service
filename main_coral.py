from core.event_handlers import ReadServiceEventHandler
from core.coral.coral_detector import CoralObjectDetector  
from core.coral.model_loader import load_model  
from common.event_bus.event_bus import EventBus
from common.utilities import logger
from core.models.detections import BaseDetector
from core.utilities import register_detect_service, EventChannels


def create_object_detector_model() -> BaseDetector:
    model, labels = load_model()
    return CoralObjectDetector(model, labels)


def main():
    register_detect_service('coral_detection_service', 'detect_service_coral-instance', 'The Coral TPU Object Detection Service®')
    detector = create_object_detector_model()
    handler = ReadServiceEventHandler(detector)

    logger.info('Coral TPU object detection service will start soon')
    event_bus = EventBus(EventChannels.snapshot_in)
    event_bus.subscribe_async(handler)


if __name__ == '__main__':
    main()
