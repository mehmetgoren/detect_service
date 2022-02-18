from common.data.service_repository import ServiceRepository
from core.object_framers import DrawObjectFramer
from core.models.object_detector_model import BaseObjectDetectorModel
from core.object_detectors.imagehash_once_detector import ImageHashOnceDetector
from core.event_handlers import ReadServiceEventHandler
from core.jetson.model_loader import load_model
from core.jetson.jetson_detector_model import JetsonbjectDetectorModel

from common.config import DeviceType
from common.data.heartbeat_repository import HeartbeatRepository
from common.event_bus.event_bus import EventBus
from common.utilities import logger, crate_redis_connection, RedisDb


def create_object_detector_model() -> BaseObjectDetectorModel:
    model = load_model()
    return JetsonbjectDetectorModel(model)


def main():
    service_name = 'jetson_detection_service'
    # noinspection DuplicatedCode
    connection = crate_redis_connection(RedisDb.MAIN)
    heartbeat = HeartbeatRepository(connection, service_name)
    heartbeat.start()

    service_repository = ServiceRepository(connection)
    service_repository.add(service_name)

    framer = DrawObjectFramer()
    detector = ImageHashOnceDetector(DeviceType.IOT, create_object_detector_model())
    handler = ReadServiceEventHandler(detector, framer)

    logger.info('jetson object detection service will start soon')
    event_bus = EventBus('read')
    event_bus.subscribe(handler)


if __name__ == '__main__':
    main()
