import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class denseFPN_121(nn.Module):
    """ DenseNet121-based Feature Pyramid Network (FPN) for feature extraction. 
        Total number of parameters:  8232320 (8.23 million) 
        Returns a feature map of size 256x12x12. """ 
    def __init__(self, weights='DEFAULT', common_channel_size=256, output_channel_size=256, output_feature_size=12):
        """
        Initializes the denseFPN_121 class.

        Args:
            weights (str): The weights to use for the denseFPN_121 Small features. Default is 'DEFAULT'.
            common_channel_size (int): The size of the common channel. Default is 256.
            output_channel_size (int): The size of the output channel. Default is 256.
            output_feature_size (int): The size of the output feature. Default is 12.
        """
        super(denseFPN_121, self).__init__()
        original_densenet = models.densenet121(weights=weights)
        
        # Initial layers: extract features without modification
        self.encoder = nn.ModuleList([
            nn.Sequential(*list(original_densenet.features.children())[:6], nn.Dropout(0.1)),   # 128x12x12
            nn.Sequential(*list(original_densenet.features.children())[6:8], nn.Dropout(0.2)),  # 256x6x6
            nn.Sequential(*list(original_densenet.features.children())[8:10], nn.Dropout(0.3)), # 896x3x3
            nn.Sequential(*list(original_densenet.features.children())[10:-1], nn.Dropout(0.3)) # 1920x3x3
        ])
        
        # Define convolutional layers for adapting channel sizes
        fpn_channels = [128, 256, 512, 1024]
        self.adaptation_layers = nn.ModuleDict({
            f'adapt{i+1}': nn.Conv2d(fpn_channels[i], common_channel_size, kernel_size=1)
            for i in range(4)
        })

        # Define FPN layers
        self.fpn = nn.ModuleDict({
            f'fpn{i+1}': nn.Conv2d(common_channel_size, common_channel_size, kernel_size=1)
            for i in range(3)
        })

        self.merge_layers = nn.Sequential(
            nn.Conv2d(common_channel_size, output_channel_size, kernel_size=1), # kernel size 1 or 3 w padding 1
            nn.BatchNorm2d(output_channel_size),
            nn.ReLU(),
            nn.Dropout(0.2) # 0.2
        )
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_feature_size)

    def forward(self, x):
        # Encoder
        features = []
        for encoder in self.encoder:
            x = encoder(x)
            features.append(x)
        
        # Merge channels using 1x1 convolutions
        adapted_features = [self.adaptation_layers[f'adapt{i+1}'](features[i]) for i in range(4)]
        
        # FPN integration using top-down pathway
        fpn_output = adapted_features.pop()  # Start with the deepest features
        for i in reversed(range(3)):
            upsampled = F.interpolate(fpn_output, size=adapted_features[i].shape[-2:], mode='nearest')
            fpn_output = self.fpn[f'fpn{i+1}'](upsampled + adapted_features[i])
        
        # Merge features
        merged_features = self.merge_layers(fpn_output)
        
        merged_features = self.global_avg_pool(merged_features)
        
        return merged_features # 256x12x12


class denseFPN_201(nn.Module):
    """ DenseNet201-based Feature Pyramid Network (FPN) for feature extraction.
        Total number of parameters:  19697280 (19.70 million) 
        Returns a feature map of size 256x12x12. """ 
    def __init__(self, weights='DEFAULT', common_channel_size=256, output_channel_size=256, output_feature_size=12):
        """
        Initializes the denseFPN_201 class.

        Args:
            weights (str): The weights to use for the EfficientNet V2 Small features. Default is 'DEFAULT'.
            common_channel_size (int): The size of the common channel. Default is 256.
            output_channel_size (int): The size of the output channel. Default is 256.
            output_feature_size (int): The size of the output feature. Default is 12.
        """
        super(denseFPN_201, self).__init__()
        original_densenet = models.densenet201(weights=weights)
        
        # Initial layers: extract features without modification
        self.encoder = nn.ModuleList([
            nn.Sequential(*list(original_densenet.features.children())[:6], nn.Dropout(0.4)),   # 128x12x12
            nn.Sequential(*list(original_densenet.features.children())[6:8], nn.Dropout(0.4)),  # 256x6x6
            nn.Sequential(*list(original_densenet.features.children())[8:10], nn.Dropout(0.4)), # 896x3x3
            nn.Sequential(*list(original_densenet.features.children())[10:-1], nn.Dropout(0.4)) # 1920x3x3
        ])
        
        # Define convolutional layers for adapting channel sizes
        fpn_channels = [128, 256, 896, 1920]
        self.adaptation_layers = nn.ModuleDict({
            f'adapt{i+1}': nn.Conv2d(fpn_channels[i], common_channel_size, kernel_size=1)
            for i in range(4)
        })

        # Define FPN layers
        self.fpn = nn.ModuleDict({
            f'fpn{i+1}': nn.Conv2d(common_channel_size, common_channel_size, kernel_size=1)
            for i in range(3)
        })

        self.merge_layers = nn.Sequential(
            nn.Conv2d(common_channel_size, output_channel_size, kernel_size=1), # kernel size 1 or 3
            nn.BatchNorm2d(output_channel_size),
            nn.ReLU(),
            nn.Dropout(0.4) # 0.2
        )
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_feature_size)

    def forward(self, x):
        # Encoder
        features = []
        for encoder in self.encoder:
            x = encoder(x)
            features.append(x)
        
        # Merge channels using 1x1 convolutions
        adapted_features = [self.adaptation_layers[f'adapt{i+1}'](features[i]) for i in range(4)]
        
        # FPN integration using top-down pathway
        fpn_output = adapted_features.pop()  # Start with the deepest features
        for i in reversed(range(3)):
            upsampled = F.interpolate(fpn_output, size=adapted_features[i].shape[-2:], mode='nearest')
            fpn_output = self.fpn[f'fpn{i+1}'](upsampled + adapted_features[i])
        
        # Merge features
        merged_features = self.merge_layers(fpn_output)
        merged_features = self.global_avg_pool(merged_features)
        return merged_features # 256x12x12


class efficientFPN_v2_s(nn.Module):
    """ EfficientNet V2 Small-based Feature Pyramid Network (FPN) for feature extraction.
        Total number of parameters:  21008208 (21.01 million) 
        Returns a feature map of size 256x25x25. """ 
    def __init__(self, weights='DEFAULT', common_channel_size=256, output_channel_size=256, output_feature_size=25):
        """
        Initializes the efficientFPN_v2_s class.

        Args:
            weights (str): The weights to use for the EfficientNet V2 Small features. Default is 'DEFAULT'.
            common_channel_size (int): The size of the common channel. Default is 256.
            output_channel_size (int): The size of the output channel. Default is 256.
            output_feature_size (int): The size of the output feature. Default is 25.
        """
        super(efficientFPN_v2_s, self).__init__()
        # Load EfficientNet V2 Small features
        efficientnet_v2_s = models.efficientnet_v2_s(weights=weights).features[:-1]

        # Modularize encoders
        self.encoders = nn.ModuleList([
            nn.Sequential(*list(efficientnet_v2_s.children())[:2], nn.Dropout(0.1)),    # 24x50x50
            nn.Sequential(*list(efficientnet_v2_s.children())[2:3], nn.Dropout(0.1)),   # 48x25x25
            nn.Sequential(*list(efficientnet_v2_s.children())[3:4], nn.Dropout(0.2)),   # 64x13x13
            nn.Sequential(*list(efficientnet_v2_s.children())[4:5], nn.Dropout(0.2)),   # 128x7x7
            nn.Sequential(*list(efficientnet_v2_s.children())[5:6], nn.Dropout(0.3)),   # 160x7x7
            nn.Sequential(*list(efficientnet_v2_s.children())[6:7], nn.Dropout(0.3))    # 256x4x4
        ])
        
        # Define convolutional layers for adapting channel sizes
        fpn_channels = [24, 48, 64, 128, 160, 256]  # example channel sizes based on architecture details
        self.adaptation_layers = nn.ModuleDict({
            f'adapt{i+1}': nn.Conv2d(fpn_channels[i], common_channel_size, kernel_size=1)
            for i in range(6)
        })

        # Define FPN layers
        self.fpn = nn.ModuleDict({
            f'fpn{i+1}': nn.Conv2d(common_channel_size, common_channel_size, kernel_size=1)
            for i in range(6)
        })

        self.merge_layers = nn.Sequential(
            nn.Conv2d(common_channel_size, output_channel_size, kernel_size=1),
            nn.BatchNorm2d(output_channel_size),
            nn.SiLU(),
            nn.Dropout(0.3)
        )
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_feature_size)

    def forward(self, x):
        # Forward pass through encoders
        features = []
        for encoder in self.encoders:
            x = encoder(x)
            features.append(x)
        
        # Merge channels using 1x1 convolutions
        adapted_features = [self.adaptation_layers[f'adapt{i+1}'](features[i]) for i in range(6)]
        
        # FPN integration using top-down pathway
        fpn_output = adapted_features.pop()  # Start with the deepest features
        for i in reversed(range(0,4)):
            upsampled = F.interpolate(fpn_output, size=adapted_features[i].shape[-2:], mode='nearest')
            fpn_output = self.fpn[f'fpn{i+1}'](upsampled + adapted_features[i])
        
        # Merge features and apply global pooling
        merged_features = self.merge_layers(fpn_output)
        pooled_features = self.global_avg_pool(merged_features)
        return pooled_features # 256x25x25


class efficientDecoder_v2_s(nn.Module):
    """ EfficientNet V2 Small-based encoder-decoder architecture for feature extraction.
        Total number of parameters:  20971120 (21.00 million) 
        Returns a feature map of size 256x25x25. """ 
    def __init__(self, weights='DEFAULT', common_channel_size=None, output_channel_size=256, output_feature_size=25):
        """
        Initializes the efficientDecoder_v2_s class.

        Args:
            weights (str): The weights to use for the EfficientNet V2 Small features. Default is 'DEFAULT'.
            common_channel_size (int): The size of the common channel. Default is None.
            output_channel_size (int): The size of the output channel. Default is 256.
            output_feature_size (int): The size of the output feature. Default is 25.
        """
        super(efficientDecoder_v2_s, self).__init__()
        # Load EfficientNet V2 Small features
        efficientnet_v2_s = models.efficientnet_v2_s(weights=weights).features[:-1]

        # Modularize encoders
        self.encoders = nn.ModuleList([
            nn.Sequential(*list(efficientnet_v2_s.children())[:2], nn.Dropout(0.1)),    # 24x50x50
            nn.Sequential(*list(efficientnet_v2_s.children())[2:3], nn.Dropout(0.1)),   # 48x25x25
            nn.Sequential(*list(efficientnet_v2_s.children())[3:4], nn.Dropout(0.2)),   # 64x13x13
            nn.Sequential(*list(efficientnet_v2_s.children())[4:5], nn.Dropout(0.2)),   # 128x7x7
            nn.Sequential(*list(efficientnet_v2_s.children())[5:6], nn.Dropout(0.3)),   # 160x7x7
            nn.Sequential(*list(efficientnet_v2_s.children())[6:7], nn.Dropout(0.3))    # 256x4x4
        ])
        
        # Modularize upconvolutions
        self.upconvs = nn.ModuleList([
            nn.ConvTranspose2d(in_channels=256, out_channels=160, kernel_size=2, stride=2, padding=1, output_padding=1),
            nn.Conv2d(in_channels=160, out_channels=128, kernel_size=1, stride=1),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(in_channels=64, out_channels=48, kernel_size=2, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(in_channels=48, out_channels=24, kernel_size=2, stride=2)
        ])
        
        # Modularize decoders
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(160*2, 160, kernel_size=3, padding=1),
                nn.BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True),
                nn.SiLU(inplace=True),
                nn.Dropout(0.3, inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(128*2, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True),
                nn.SiLU(inplace=True),
                nn.Dropout(0.3, inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(64*2, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True),
                nn.SiLU(inplace=True),
                nn.Dropout(0.2, inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(48*2, 48, kernel_size=3, padding=1),
                nn.BatchNorm2d(48, eps=0.001, momentum=0.1, affine=True, track_running_stats=True),
                nn.SiLU(inplace=True),
                nn.Dropout(0.2, inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(24*2, 24, kernel_size=3, padding=1),
                nn.BatchNorm2d(24, eps=0.001, momentum=0.1, affine=True, track_running_stats=True),
                nn.SiLU(inplace=True),
                nn.Dropout(0.1, inplace=True)
            )
        ])

        # Optional, merge layers to increase the number of channels
        self.merge_layers = nn.Sequential(
            nn.Conv2d(24, output_channel_size, kernel_size=1),
            nn.BatchNorm2d(output_channel_size),
            nn.SiLU(),
            nn.Dropout(0.3)
        )
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_feature_size) # to reduce noise and overfitting

    def forward(self, x):
        # Encoder
        features = []
        for encoder in self.encoders:
            x = encoder(x)
            features.append(x)
        
        # Decoder
        x = features.pop()
        for upconv, decoder, feature in zip(self.upconvs, self.decoders, reversed(features)):
            x = upconv(x)
            x = torch.cat((x, feature), dim=1)
            x = decoder(x)
        
        x = self.merge_layers(x) # Introduced to increase the number of channels
        pooled_features = self.global_avg_pool(x) # Introduced to reduce noise and overfitting
        
        return pooled_features # 256x25x25
    