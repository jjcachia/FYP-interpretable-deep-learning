import torch
import torch.nn as nn
from torchvision import models
from src.models.backbone_models import denseNet121, denseNet169, denseNet201, resNet34, resNet152, vgg16, vgg19

# Dictionary of supported backbone models
BACKBONE_DICT = {
    'denseNet121': denseNet121,
    'denseNet169': denseNet169,
    'denseNet201': denseNet201,
    'resNet34': resNet34,
    'resNet152': resNet152,
    'vgg16': vgg16,
    'vgg19': vgg19
}


############################################################################################################################################################################
################################################################### Base Model for Malignancy Prediction ###################################################################
############################################################################################################################################################################

class BaseModel(nn.Module):
    def __init__(self, backbone, weights, common_channel_size, hidden_layers):
        super(BaseModel, self).__init__()        
        self.backbone = backbone(weights=weights, common_channel_size=common_channel_size)
        
        # cnn_backbone_output_channel_size = self.backbone.get_output_channels()
        out_C, out_H, out_W = self.backbone.get_output_dims()
        
        # self.add_on_layers = nn.Sequential(
        #     nn.Conv2d(in_channels=out_C, out_channels=512, kernel_size=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        #     nn.Dropout(0.2)
        # )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.final_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_C, hidden_layers),
            nn.BatchNorm1d(hidden_layers),
            nn.ReLU(),
            nn.Dropout(0.2), # 0.2
            nn.Linear(hidden_layers, 1)
        )
        
    def forward(self, x):
        # Feature Extraction
        x = self.backbone(x)
        
        # x = self.add_on_layers(x)
        x = self.adaptive_pool(x)
        
        # Final Malignancy Prediction
        final_output = torch.sigmoid(self.final_classifier(x))
        
        return final_output


def construct_baseModel(backbone_name='denseNet_121', weights='DEFAULT', common_channel_size=None, hidden_layers=256):
    """
    Constructs a base model for Malignancy Prediction.

    Args:
        backbone_name (str): Name of the backbone model. Default is 'densetFPN_121'.
        weights (str): Weights to be used for the model. Default is 'DEFAULT'.
        input_dim (tuple): Dimensions of the input data. Default is (256, 12, 12).
        hidden_layers (int): Number of hidden layers in the model. Default is 1024.

    Returns:
        BaseModel: The constructed base model.

    Raises:
        ValueError: If the specified backbone_name is not supported.
    """
    if backbone_name not in BACKBONE_DICT:
        raise ValueError(f"Unsupported model name {backbone_name}")
    backbone = BACKBONE_DICT[backbone_name]
    return BaseModel(backbone=backbone, weights=weights, common_channel_size=common_channel_size, hidden_layers=hidden_layers)
