from common.data.service_repository import ServiceRepository
from core.object_framers import DrawObjectFramer
from core.object_detectors.imagehash_once_detector import ImageHashOnceDetector
from core.event_handlers import ReadServiceEventHandler

from common.config import DeviceType
from common.data.heartbeat_repository import HeartbeatRepository
from common.event_bus.event_bus import EventBus
from common.utilities import logger, crate_redis_connection, RedisDb
from core.tf.tf_object_detector_model import TfObjectDetectorModel


def register_detect_service():
    connection_service = crate_redis_connection(RedisDb.MAIN)
    service_name = 'tensorflow_detection_service'
    heartbeat = HeartbeatRepository(connection_service, service_name)
    heartbeat.start()
    service_repository = ServiceRepository(connection_service)
    service_repository.add(service_name, 'The Tensorflow Object Detection Service®')


def main():
    register_detect_service()

    detector_model = TfObjectDetectorModel()
    framer = DrawObjectFramer()
    detector = ImageHashOnceDetector(DeviceType.PC, detector_model)
    handler = ReadServiceEventHandler(detector, framer)

    logger.info('tensorflow object detection service will start soon')
    event_bus = EventBus('read_service')
    event_bus.subscribe_async(handler)


if __name__ == '__main__':
    main()
