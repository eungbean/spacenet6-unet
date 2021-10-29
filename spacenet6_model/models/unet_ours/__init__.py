from .unet_model import UNet
import torch
from .unet_model import UNet

def get_unet(config):


    model = UNet(
        n_channels=config.MODEL.IN_CHANNELS, 
        n_classes=len(config.INPUT.CLASSES),
        bilinear=True
        )

    model = torch.nn.DataParallel(model)

    if config.MODEL.WEIGHT and config.MODEL.WEIGHT != 'none':
        # load weight from file
        model.load_state_dict(
            torch.load(
                config.MODEL.WEIGHT,
                map_location=torch.device('cpu')
            )
        )

    model = model.to(config.MODEL.DEVICE)
    return model