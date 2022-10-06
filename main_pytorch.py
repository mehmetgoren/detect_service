from core.event_handlers import ReadServiceEventHandler
from core.models.detections import BaseDetector
from core.pytorch.model_loader import load_model
from common.event_bus.event_bus import EventBus
from common.utilities import logger
from core.pytorch.yolov5_object_detector import Yolov5ObjectDetector
from core.utilities import register_detect_service, EventChannels


def create_object_detector_model() -> BaseDetector:
    model = load_model()
    return Yolov5ObjectDetector(model)


def main():
    register_detect_service('pytorch_detection_service', 'detect_service_pytorch-instance', 'The PyTorch Object Detection Service®')
    detector = create_object_detector_model()
    handler = ReadServiceEventHandler(detector)

    logger.info('pytorch object detection service will start soon')
    event_bus = EventBus(EventChannels.snapshot_in)
    event_bus.subscribe_async(handler)


if __name__ == '__main__':
    main()
