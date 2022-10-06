from enum import Enum

from common.data.heartbeat_repository import HeartbeatRepository
from common.data.service_repository import ServiceRepository
from common.utilities import crate_redis_connection, RedisDb


def register_detect_service(service_name: str, instance_name: str, description: str):
    connection_main = crate_redis_connection(RedisDb.MAIN)
    heartbeat = HeartbeatRepository(connection_main, service_name)
    heartbeat.start()
    service_repository = ServiceRepository(connection_main)
    service_repository.add(service_name, instance_name, description)
    return connection_main


class EventChannels(str, Enum):
    snapshot_in = 'snapshot_in'
    snapshot_out = 'snapshot_out'
    od_service = 'od_service'
