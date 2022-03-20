from core.object_framers import DrawObjectFramer
from core.object_detectors.imagehash_once_detector import ImageHashOnceDetector
from core.event_handlers import ReadServiceEventHandler
from common.event_bus.event_bus import EventBus
from common.utilities import logger
from core.tf.tf_object_detector_model import TfObjectDetectorModel
from core.utilities import listen_data_changed_event, register_detect_service


def main():
    conn = register_detect_service('tensorflow_detection_service', 'The Tensorflow Object Detection Service®')
    listen_data_changed_event(conn)

    detector_model = TfObjectDetectorModel()
    framer = DrawObjectFramer()
    detector = ImageHashOnceDetector(conn, detector_model)
    handler = ReadServiceEventHandler(detector, framer)

    logger.info('tensorflow object detection service will start soon')
    event_bus = EventBus('read_service')
    event_bus.subscribe_async(handler)


if __name__ == '__main__':
    main()
