from core.event_handlers import ReadServiceEventHandler
from common.event_bus.event_bus import EventBus
from common.utilities import logger
from core.tf.tf_object_detector import TfObjectDetector
from core.utilities import register_detect_service, EventChannels


def main():
    register_detect_service('tensorflow_detection_service', 'detect_service_tf-instance', 'The Tensorflow Object Detection Service®')

    detector = TfObjectDetector()
    handler = ReadServiceEventHandler(detector)

    logger.info('tensorflow object detection service will start soon')
    event_bus = EventBus(EventChannels.snapshot_in)
    event_bus.subscribe_async(handler)


if __name__ == '__main__':
    main()
