import jetson.inference as inf

from common.utilities import config


def load_model():
    name = config.jetson.model_name
    net = inf.detectNet(name, threshold=.1)
    return net
