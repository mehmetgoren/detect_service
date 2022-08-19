from redis.client import Redis

from common.data.source_repository import SourceRepository
from common.utilities import logger
from core.data.od import Od
from core.data.od_model import OdModel
from core.data.od_repository import OdRepository


class OdCache:
    models = {}

    def __init__(self, connection: Redis):
        self.od_repository = OdRepository(connection)
        self.source_repository = SourceRepository(connection)

    def get_od_model(self, detected_by: str) -> Od:
        if detected_by not in OdCache.models:
            od_model = self.od_repository.get(detected_by)
            if od_model is None:
                source_model = self.source_repository.get(detected_by)
                if source_model is None:
                    logger.warn(f'source was not found for Object Detection Model, Detection will not work for {detected_by}')
                    return None
                od_model = OdModel().map_from(source_model)
                self.od_repository.add(od_model)
            OdCache.models[detected_by] = Od().map_from(od_model)
        return OdCache.models[detected_by]

    def refresh(self, detected_by: str) -> Od:
        if detected_by in OdCache.models:
            del OdCache.models[detected_by]
        return self.get_od_model(detected_by)

    @staticmethod
    def remove(detected_by: str):
        OdCache.models[detected_by] = None
