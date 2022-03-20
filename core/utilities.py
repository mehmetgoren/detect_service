import time
from threading import Thread
from redis.client import Redis

from common.data.heartbeat_repository import HeartbeatRepository
from common.data.service_repository import ServiceRepository
from common.event_bus.event_bus import EventBus
from common.utilities import logger, crate_redis_connection, RedisDb
from core.event_handlers import DataChangedEventHandler


def register_detect_service(service_name: str, description: str):
    connection_main = crate_redis_connection(RedisDb.MAIN)
    heartbeat = HeartbeatRepository(connection_main, service_name)
    heartbeat.start()
    service_repository = ServiceRepository(connection_main)
    service_repository.add(service_name, description)
    return connection_main


def listen_data_changed_event(connection: Redis):
    def fn():
        while 1:
            event_bus = None
            try:
                handler = DataChangedEventHandler(connection)
                event_bus = EventBus('data_changed')
                event_bus.subscribe_async(handler)
            except BaseException as ex:
                logger.error(f'an error occurred on listen data changed event, ex: {ex}')
            finally:
                if event_bus is not None:
                    try:
                        event_bus.unsubscribe()
                    except BaseException as ex:
                        logger.error(f'an error occurred during the unsubscribing data changed event, err: {ex}')
            time.sleep(1.)

    th = Thread(target=fn)
    th.daemon = False
    th.start()
