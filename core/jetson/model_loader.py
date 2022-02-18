import jetson.inference as inf

from common.utilities import config


def load_model():
    name = config.jetson.model_name
    threshold = config.jetson.threshold
    net = inf.detectNet(name, threshold=threshold)
    return net
