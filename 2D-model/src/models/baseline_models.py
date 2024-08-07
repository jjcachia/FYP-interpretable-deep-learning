import torch
import torch.nn as nn
from torchvision import models
from src.models.backbone_models import densetFPN_121, densetFPN_201, efficientFPN_v2_s, efficientDecoder_v2_s

# Define the dictionary outside the class
BACKBONE_DICT = {
    'denseNet121': models.densenet121(weights='DEFAULT').features[:-1],
    'denseNet201': models.densenet201(weights='DEFAULT').features[:-1],
    'efficientNetV2_s': models.efficientnet_v2_s(weights='DEFAULT').features[:-1],
    'densetFPN_121': densetFPN_121,
    'densetFPN_201': densetFPN_201,
    'efficientFPN_v2_s': efficientFPN_v2_s,
    'efficientDecoder_v2_s': efficientDecoder_v2_s
}


class BaseModel(nn.Module):
    """Base Model for Malignancy Prediction."""
    def __init__(self, backbone, weights='DEFAULT', input_dim=(128,12,12), hidden_layers=1024):
        super(BaseModel, self).__init__()

        self.backbone = backbone(weights=weights)
        
        input_channels, input_width, input_height = input_dim
        
        self.final_classifier = nn.Sequential(
            nn.Linear(input_channels*input_width*input_height, hidden_layers),
            nn.BatchNorm1d(hidden_layers),
            nn.ReLU(),
            nn.Dropout(0.1), # 0.2
            nn.Linear(hidden_layers, 1)
        )
        
    def forward(self, x):
        # Feature Extraction
        x = self.backbone(x)
        
        # Final Malignancy Prediction
        final_output = torch.sigmoid(self.final_classifier(x))
        
        return final_output

def construct_baseModel(backbone_name='densetFPN_121', weights='DEFAULT', input_dim=(256,12,12), hidden_layers=1024):
    if backbone_name not in BACKBONE_DICT:
        raise ValueError(f"Unsupported model name {backbone_name}")
    backbone = BACKBONE_DICT[backbone_name]
    return BaselineModel(backbone=backbone, weights=weights, input_dim=input_dim, hidden_layers=hidden_layers)


class BaselineModel(nn.Module):
    """ Hierarchical Multi-Task Learning Model with shared feature extraction and task-specific 
        classifiers for nodule characteristics, and final malignancy prediction."""
    def __init__(self, backbone, weights, input_dim=(256,25,25), hidden_layers=256, num_tasks=5):
        super(BaselineModel, self).__init__()

        self.backbone = backbone(weights=weights)
        
        input_channels, input_width, input_height = input_dim
        
        self.task_specific_layers = nn.ModuleList([
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_channels*input_width*input_height, hidden_layers), # was 1024, lowering it makes it more faithful to interpretation
                nn.BatchNorm1d(hidden_layers),
                nn.ReLU(),
                nn.Dropout(0.2) # 0.2
            ) for _ in range(num_tasks)
        ])
        
        self.task_specific_classifier = nn.ModuleList([
            nn.Linear(hidden_layers, 1) for _ in range(num_tasks)
        ])
        
        self.final_classifier = nn.Sequential(
            nn.Linear(hidden_layers * num_tasks, hidden_layers),
            nn.BatchNorm1d(hidden_layers),
            nn.ReLU(),
            nn.Dropout(0.1), # 0.2
            nn.Linear(hidden_layers, 1)
        )
        
    def forward(self, x):
        # Feature Extraction
        x = self.backbone(x)
        
        # Process intermediate outputs
        intermediate_outputs = [layer(x) for layer in self.task_specific_layers]
        
        # Concatenate intermediate outputs
        concatenated_outputs = torch.cat(intermediate_outputs, dim=1)
        
        # Nodule Characteristics Prediction
        task_outputs = [torch.sigmoid(self.task_specific_classifier[i](intermediate_outputs[i])) for i in range(len(intermediate_outputs))]
        
        # Final Malignancy Prediction
        final_output = torch.sigmoid(self.final_classifier(concatenated_outputs))
        
        return final_output, task_outputs
    
def construct_baselineModel(backbone_name, weights, input_dim=(256,25,25), hidden_layers=256, num_tasks=5):
    if backbone_name not in BACKBONE_DICT:
        raise ValueError(f"Unsupported model name {backbone_name}")
    backbone = BACKBONE_DICT[backbone_name]
    return BaselineModel(backbone=backbone, weights=weights, input_dim=input_dim, hidden_layers=hidden_layers, num_tasks=num_tasks)
