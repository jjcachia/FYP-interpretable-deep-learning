import torch
import torch.nn as nn
from torchvision import models
from src.models.backbone_models import denseFPN_121, denseFPN_201, efficientFPN_v2_s, efficientDecoder_v2_s

# Dictionary of supported backbone models
BACKBONE_DICT = {
    'denseNet121': models.densenet121(weights='DEFAULT').features[:-1],
    'denseNet201': models.densenet201(weights='DEFAULT').features[:-1],
    'efficientNetV2_s': models.efficientnet_v2_s(weights='DEFAULT').features[:-1],
    'denseFPN_121': denseFPN_121,
    'denseFPN_201': denseFPN_201,
    'efficientFPN_v2_s': efficientFPN_v2_s,
    'efficientDecoder_v2_s': efficientDecoder_v2_s
}


############################################################################################################################################################################
##### Hierarchical Multi-Task Learning Model with shared feature extraction and task-specific classifiers for nodule characteristics, and final malignancy prediction. #####
############################################################################################################################################################################

class BaselineModel(nn.Module):
    def __init__(self, backbone, weights, common_channel_size, output_channel_size, output_feature_size, hidden_layers, num_tasks):
        super(BaselineModel, self).__init__()
        self.backbone = backbone(weights=weights, common_channel_size=common_channel_size, output_channel_size=output_channel_size, output_feature_size=output_feature_size)
        
        self.task_specific_layers = nn.ModuleList([
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(common_channel_size*output_feature_size*output_feature_size, hidden_layers), # was 1024, lowering it makes it more faithful to interpretation
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
    
    
def construct_baselineModel(backbone_name='efficientFPN_v2_s', 
                            weights='DEFAULT', 
                            common_channel_size=256, 
                            output_channel_size=256, 
                            output_feature_size=25, 
                            hidden_layers=256, 
                            num_tasks=5):
    """
    Constructs a Hierarchical Multi-Task Learning Baseline Model.

    Args:
        backbone_name (str): Name of the backbone model.
        weights (str): Weights to initialize the model with. Default is 'DEFAULT'.
        common_channel_size (int): Size of the common channel in the model. Default is 256.
        output_channel_size (int): Size of the output channel in the model. Default is 256.
        output_feature_size (int): Size of the output feature in the model. Default is 25.
        hidden_layers (int): Number of hidden layers in the model. Default is 256.
        num_tasks (int): Number of tasks for the model. Default is 5.

    Returns:
        BaselineModel: The constructed baseline model.

    Raises:
        ValueError: If the specified backbone name is not supported.
    """
    
    if backbone_name not in BACKBONE_DICT:
        raise ValueError(f"Unsupported model name {backbone_name}")
    backbone = BACKBONE_DICT[backbone_name]
    
    return BaselineModel(backbone=backbone, 
                         weights=weights, 
                         common_channel_size=common_channel_size, 
                         output_channel_size=output_channel_size, 
                         output_feature_size=output_feature_size,
                         hidden_layers=hidden_layers, 
                         num_tasks=num_tasks)

