from shapely.geometry import Polygon

from core.data.od_model import OdModel
from core.models.object_detector_model import DetectionBox


class Od:
    def __init__(self):
        self.id: str = ''
        self.brand: str = ''
        self.name: str = ''
        self.address: str = ''
        self.created_at: str = ''
        self.selected_list_length: int = 0
        self.selected_list: dict = {}
        self.zone_list: Polygon = Polygon([])
        self.separator = 'º'

    @staticmethod
    def __create_area(do: DetectionBox):
        x1, y1, x2, y2 = do.x1, do.y1, do.x2, do.y2
        return Polygon([(x1, y1), (x1, y2), (x2, y1), (x2, y2)])

    @staticmethod
    def __create_polygon(value: str, separator: str):
        if len(value) == 0:
            return Polygon([])
        arr = value.split(separator)  # arr.length is always an even number
        length = int(len(arr) / 2)
        index = 0
        points = []
        for j in range(length):
            x = float(arr[index])
            index += 1
            y = float(arr[index])
            index += 1
            points.append((x, y))
        return Polygon(points)

    def is_in_zone(self, do: DetectionBox):
        if self.zone_list.length == 0:
            return True
        return self.zone_list.intersects(self.__create_area(do))

    def is_selected(self, cls_idx: int):
        return cls_idx in self.selected_list

    def check_threshold(self, cls_idx: int, threshold: float):
        return threshold >= self.selected_list[cls_idx]

    def map_from(self, od_model: OdModel):
        self.id = od_model.id
        self.brand = od_model.brand
        self.name = od_model.name
        self.address = od_model.address
        self.created_at = od_model.created_at
        indices = [int(item) for item in od_model.selected_list.split(self.separator)] if len(od_model.selected_list) > 0 else []
        self.selected_list_length = len(indices)
        if self.selected_list_length > 0:
            thresholds = od_model.threshold_list.split(self.separator)  # thresholds.length must be equal to indices' length
            index = 0
            for cls_idx in indices:
                self.selected_list[cls_idx] = float(thresholds[index])
                index += 1
        self.zone_list = self.__create_polygon(od_model.zone_list, self.separator)
        return self
