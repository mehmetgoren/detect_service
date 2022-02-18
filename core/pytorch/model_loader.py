import torch

from common.utilities import config


def load_model():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # torch.device('cpu')
    half = device.type != 'cpu'

    model = torch.hub.load(config.torch.model_name, config.torch.model_name_specific,
                           pretrained=True)  # force_reload=True)

    model.to(device)
    if half:
        model.half()  # to FP16
    model.eval()
    # torch.backends.cudnn.benchmark = True
    # torch.cuda.empty_cache()
    return model

#
# import torchvision.models as models
# def load_model_mobile():
#     model = models.alexnet() #torch.hub.load('pytorch/vision:v0.9.0', 'deeplabv3_resnet101', pretrained=True)
#     model.to(device)
#     if half:
#         model.half()  # to FP16
#     model.eval()
#     model.
#     return model
