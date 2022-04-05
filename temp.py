import datetime
import json

from shapely.geometry import Polygon


def in_area():
    # p = Polygon([(1, 1), (2, 2), (4, 2), (3, 1)])
    # q = Polygon([(1.5, 2), (3, 5), (5, 4), (3.5, 1)])
    # # print(p.intersects(q))  # True
    # # print(p.intersection(q).area)  # 1.0
    # # x = p.intersection(q)
    # # print(x)

    # image_corr = Polygon([(45, 45), (45, 60), (60, 60), (60, 45)])
    # zones = Polygon([(0, 0), (0, 5), (5, 5), (5, 0)])  # Polygon([(5, 5), (5, 95), (95, 95), (95, 5)])
    # print(image_corr.intersects(zones))
    # print(zones.intersects(image_corr))
    # print(image_corr.intersection(zones).area)

    image_corr = Polygon([(0, 0), (0, 720), (1280, 720), (0, 1280)])
    zones = Polygon([(410, 0), (454, 635), (21, 639), (0, 0)])
    print(image_corr.intersects(zones))

    image_corr = Polygon([(0, 0), (0, 720), (1280, 720), (0, 1280)])
    zones = Polygon([(778, 612), (1095, 610), (1117, 162), (783, 158)])
    print(image_corr.intersects(zones))

    start = datetime.datetime.now()
    length = 1000000
    for j in range(length):
        result = image_corr.intersects(zones)
        # print(source.name)
    end = datetime.datetime.now()

    print(f'result: {(end - start).seconds}')


class Test:
    models = {}

    def __init__(self):
        self.od_repository = {}

    def set(self, source_id: str):
        self.od_repository[source_id] = source_id


def execute_test():
    t = Test()
    t.set('t1')
    t.models['models'] = 'model'

    t2 = Test()
    t2.set('t2')

    print(t.od_repository)
    print(t2.od_repository)
    print(Test.models)


detectedObjects = [{'cls_index': 0, 'cls_name': 'person', 'score': 0.78}, {'cls_index': 1, 'cls_name': 'traffic_light', 'score': 0.43}]

dic = {'file_name': 'Tapo 200 (2)_traffic light_0.41_2022_02_23_02_18_32_206_1fc616a7d1204e158799a6a1071fa6b1',
       'video_clip_enabled': True, 'detected_objects': detectedObjects}

print(json.dumps(dic))
