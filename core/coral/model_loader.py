from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.dataset import read_label_file

from common.utilities import config


def load_model():
    path = config.coral.model_path
    interpreter = make_interpreter(path)
    interpreter.allocate_tensors()
    labels = read_label_file(config.coral.labels_path) if config.coral.labels_path else {}
    return interpreter, labels
