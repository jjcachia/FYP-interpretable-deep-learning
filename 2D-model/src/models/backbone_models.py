import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from efficientnet_pytorch_3d import EfficientNet3D

############################################################################################################################################################################
############################################################################ Base CNN Networks #############################################################################
############################################################################################################################################################################

class denseNet121(nn.Module):
    """ DenseNet121-based Feature Extractor for feature extraction.
        Total number of parameters:  6953856 (6.95 million)
        Returns a feature map of size 1024x3x3. """
    def __init__(self, weights='DEFAULT', common_channel_size=None):
        """ 
        Initializes the denseNet121 class. 
            
        Args:
            weights (str): The weights to use for the DenseNet121 features. Default is 'DEFAULT'.
            common_channel_size (int): The size of the common channel. Default is None.
        """
        super(denseNet121, self).__init__()
        densenet = models.densenet121(weights=weights)
        self.features = densenet.features  

    def forward(self, x):
        return self.features(x)
    
    #def get_output_channels(self):
    #    """ Returns the number of output channels from the final convolutional layer. """
    #    final_bn_layer = [layer for layer in self.features.modules() if isinstance(layer, nn.BatchNorm2d)][-1]
    #    final_conv_layer = [layer for layer in self.features.modules() if isinstance(layer, nn.Conv2d)][-1]
    #    return final_bn_layer.weight.shape[0]
    
    def get_output_dims(self):
        return 1024, 3, 3
    
    def conv_info(self):
        """ Returns a list of dicts containing kernel sizes, strides, and paddings for each convolutional layer. """
        conv_layers = [layer for layer in self.features.modules() if isinstance(layer, nn.Conv2d)]
        kernel_sizes, strides, paddings = [], [], []
        for conv in conv_layers:
            kernel_sizes.append(conv.kernel_size[0])
            strides.append(conv.stride[0])
            paddings.append(conv.padding[0])
        return kernel_sizes, strides, paddings

class denseNet169(nn.Module):
    """ DenseNet169-based Feature Extractor for feature extraction.
        Total number of parameters:  12688648 (12.69 million) 
        Returns a feature map of size 1664x3x3. """
    def __init__(self, weights='DEFAULT', common_channel_size=None):
        """
        Initializes the denseNet169 class.
        
        Args:
            weights (str): The weights to use for the DenseNet169 features. Default is 'DEFAULT'.
            common_channel_size (int): The size of the common channel. Default is None.
        """
        super(denseNet169, self).__init__()
        densenet = models.densenet169(weights=weights)
        self.features = densenet.features

    def forward(self, x):
        return self.features(x)
    
    def get_output_dims(self):
        """ Returns the number of output channels from the final convolutional layer. """
        return 1664, 3, 3

class denseNet201(nn.Module):
    """ DenseNet201-based Feature Extractor for feature extraction.
        Total number of parameters:  18092928 (18.09 million) 
        Returns a feature map of size 1920x3x3. """
    def __init__(self, weights='DEFAULT', common_channel_size=None):
        """ 
        Initializes the denseNet201 class.
        
        Args:
            weights (str): The weights to use for the DenseNet201 features. Default is 'DEFAULT'.
            common_channel_size (int): The size of the common channel. Default is None.
        """
        super(denseNet201, self).__init__()
        densenet = models.densenet201(weights=weights)
        self.features = densenet.features

    def forward(self, x):
        return self.features(x)
    
    def get_output_dims(self):
        """ Returns the number of output channels from the final convolutional layer. """
        return 1920, 3, 3
    
class resNet34(nn.Module):
    """ ResNet34-based Feature Extractor for feature extraction.
        Total number of parameters:  21797672 (21.80 million) 
        Returns a feature map of size 512x7x7. """
    def __init__(self, weights='DEFAULT', common_channel_size=None):
        super(resNet34, self).__init__()
        resnet = models.resnet34(weights=weights)
        self.features = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):
        return self.features(x)
    
    def get_output_dims(self):
        """ Returns the number of output channels from the final convolutional layer. """
        return 512, 4, 4

class resNet152(nn.Module):
    """ ResNet152-based Feature Extractor for feature extraction.
        Total number of parameters:  60192808 (60.19 million) 
        Returns a feature map of size 2048x7x7. """
    def __init__(self, weights='DEFAULT', common_channel_size=None):
        super(resNet152, self).__init__()
        resnet = models.resnet152(weights=weights)
        self.features = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):
        return self.features(x)
    
    def get_output_dims(self):
        """ Returns the number of output channels from the final convolutional layer. """
        return 2048, 4, 4

class vgg16(nn.Module):
    """ VGG16-based Feature Extractor for feature extraction.
        Total number of parameters:  138357544 (138.36 million) 
        Returns a feature map of size 512x7x7. """
    def __init__(self, weights='DEFAULT', common_channel_size=None):
        super(vgg16, self).__init__()
        vgg = models.vgg16(weights=weights)
        self.features = nn.Sequential(*list(vgg.children())[:-1])

    def forward(self, x):
        return self.features(x)
    
    def get_output_dims(self):
        """ Returns the number of output channels from the final convolutional layer. """
        return 512, 7, 7
    
class vgg19(nn.Module):
    """ VGG19-based Feature Extractor for feature extraction.
        Total number of parameters:  143667240 (143.67 million) 
        Returns a feature map of size 512x7x7. """
    def __init__(self, weights='DEFAULT', common_channel_size=None):
        super(vgg19, self).__init__()
        vgg = models.vgg19(weights=weights)
        self.features = nn.Sequential(*list(vgg.children())[:-1])

    def forward(self, x):
        return self.features(x)
    
    def get_output_dims(self):
        """ Returns the number of output channels from the final convolutional layer. """
        return 512, 7, 7

############################################################################################################################################################################
########################################################################## Base 3D CNN Networks ############################################################################
############################################################################################################################################################################

class efficientNet3D(nn.Module):
    """ EfficientNet3D-based Feature Extractor for feature extraction.
        Total number of parameters:  10285384 (10.29 million) 
        Returns a feature map of size 1280x4x4. """
    def __init__(self, weights='DEFAULT', common_channel_size=None):
        """
        Initializes the efficientNet3D class.
        
        Args:
            weights (str): The weights to use for the EfficientNet3D features. Default is 'DEFAULT'.
            common_channel_size (int): The size of the common channel. Default is None.
        """
        super(efficientNet3D, self).__init__()
        efficientNet = EfficientNet3D.from_name("efficientnet-b0", override_params={'num_classes': 1}, in_channels=1)
        self.features = efficientNet.extract_features

    def forward(self, x):
        return self.features(x)
    
    def get_output_dims(self):
        """ Returns the number of output channels from the final convolutional layer. """
        return 1280, 2, 2, 2


############################################################################################################################################################################
######################################################################### Feature Pyramid Networks #########################################################################
############################################################################################################################################################################

class denseFPN_121(nn.Module):
    """ DenseNet121-based Feature Pyramid Network (FPN) for feature extraction. 
        Total number of parameters:  8232320 (8.23 million) 
        Returns a feature map of size 256x12x12. """ 
    def __init__(self, weights='DEFAULT', common_channel_size=256):
        """
        Initializes the denseFPN_121 class.

        Args:
            weights (str): The weights to use for the denseFPN_121 features. Default is 'DEFAULT'.
            common_channel_size (int): The size of the FPN common channel. Default is 256.
        """
        super(denseFPN_121, self).__init__()
        original_densenet = models.densenet121(weights=weights)
        self.common_channel_size = common_channel_size
        
        # Initial layers: extract features without modification
        self.encoder = nn.ModuleList([
            nn.Sequential(*list(original_densenet.features.children())[:6], nn.Dropout(0.1)),   # 128x12x12
            nn.Sequential(*list(original_densenet.features.children())[6:8], nn.Dropout(0.2)),  # 256x6x6
            nn.Sequential(*list(original_densenet.features.children())[8:10], nn.Dropout(0.3)), # 896x3x3
            nn.Sequential(*list(original_densenet.features.children())[10:], nn.Dropout(0.3))   # 1920x3x3
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
        
        return fpn_output # 256x12x12
    
    def get_output_dims(self):
        """ Returns the number of output channels from the final convolutional layer. """
        return self.common_channel_size, 12, 12

    def conv_info(self):
        """
        Returns a list of dicts containing kernel sizes, strides, and paddings for each convolutional layer.
        This function will gather information from both the DenseNet backbone and the FPN custom layers.
        """
        kernel_sizes, strides, paddings = [], [], []

        # Traverse the encoder layers which are wrapped in Sequential blocks
        for seq in self.encoder:
            for layer in seq.modules():
                if isinstance(layer, nn.Conv2d):
                    kernel_sizes.append(layer.kernel_size[0])
                    strides.append(layer.stride[0])
                    paddings.append(layer.padding[0])

        # Check adaptation layers
        for layer in self.adaptation_layers.modules():
            if isinstance(layer, nn.Conv2d):
                kernel_sizes.append(layer.kernel_size[0])
                strides.append(layer.stride[0])
                paddings.append(layer.padding[0])

        # FPN layers (each one is a single Conv2d in a ModuleList)
        for layer in self.fpn.modules():
            if isinstance(layer, nn.Conv2d):
                kernel_sizes.append(layer.kernel_size[0])
                strides.append(layer.stride[0])
                paddings.append(layer.padding[0])

        return kernel_sizes, strides, paddings

class denseFPN_201(nn.Module):
    """ DenseNet201-based Feature Pyramid Network (FPN) for feature extraction.
        Total number of parameters:  19697280 (19.70 million) 
        Returns a feature map of size 256x12x12. """ 
    def __init__(self, weights='DEFAULT', common_channel_size=256):
        """
        Initializes the denseFPN_201 class.

        Args:
            weights (str): The weights to use for the EfficientNet V2 Small features. Default is 'DEFAULT'.
            common_channel_size (int): The size of the common channel. Default is 256.
        """
        super(denseFPN_201, self).__init__()
        original_densenet = models.densenet201(weights=weights)
        self.common_channel_size = common_channel_size
        
        # Initial layers: extract features without modification
        self.encoder = nn.ModuleList([
            nn.Sequential(*list(original_densenet.features.children())[:6], nn.Dropout(0.1)),   # 128x12x12
            nn.Sequential(*list(original_densenet.features.children())[6:8], nn.Dropout(0.2)),  # 256x6x6
            nn.Sequential(*list(original_densenet.features.children())[8:10], nn.Dropout(0.4)), # 896x3x3
            nn.Sequential(*list(original_densenet.features.children())[10:], nn.Dropout(0.4))   # 1920x3x3
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
        
        return fpn_output # 256x12x12
    
    def get_output_dims(self):
        """ Returns the number of output channels from the final convolutional layer. """
        # final_conv_layer = [layer for layer in self.fpn[-1].modules() if isinstance(layer, nn.Conv2d)][-1]
        return self.common_channel_size, 12, 12
    
    def conv_info(self):
        """
        Returns a list of dicts containing kernel sizes, strides, and paddings for each convolutional layer.
        This function will gather information from both the DenseNet backbone and the FPN custom layers.
        """
        kernel_sizes, strides, paddings = [], [], []

        # Traverse the encoder layers which are wrapped in Sequential blocks
        for seq in self.encoder:
            for layer in seq.modules():
                if isinstance(layer, nn.Conv2d):
                    kernel_sizes.append(layer.kernel_size[0])
                    strides.append(layer.stride[0])
                    paddings.append(layer.padding[0])

        # Check adaptation layers
        for layer in self.adaptation_layers.modules():
            if isinstance(layer, nn.Conv2d):
                kernel_sizes.append(layer.kernel_size[0])
                strides.append(layer.stride[0])
                paddings.append(layer.padding[0])

        # FPN layers (each one is a single Conv2d in a ModuleList)
        for layer in self.fpn.modules():
            if isinstance(layer, nn.Conv2d):
                kernel_sizes.append(layer.kernel_size[0])
                strides.append(layer.stride[0])
                paddings.append(layer.padding[0])

        return kernel_sizes, strides, paddings


class efficientFPN_v2_s(nn.Module):
    """ EfficientNet V2 Small-based Feature Pyramid Network (FPN) for feature extraction.
        Total number of parameters:  21008208 (21.01 million) 
        Returns a feature map of size 256x25x25. """ 
    def __init__(self, weights='DEFAULT', common_channel_size=256):
        """
        Initializes the efficientFPN_v2_s class.

        Args:
            weights (str): The weights to use for the EfficientNet V2 Small features. Default is 'DEFAULT'.
            common_channel_size (int): The size of the common channel. Default is 256.
        """
        super(efficientFPN_v2_s, self).__init__()
        self.common_channel_size = common_channel_size
        
        # Load EfficientNet V2 Small features
        efficientnet_v2_s = models.efficientnet_v2_s(weights=weights).features[:-1]

        # Modularize encoders
        self.encoder= nn.ModuleList([
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

    def forward(self, x):
        # Forward pass through encoders
        features = []
        for encoder in self.encoder:
            x = encoder(x)
            features.append(x)
        
        # Merge channels using 1x1 convolutions
        adapted_features = [self.adaptation_layers[f'adapt{i+1}'](features[i]) for i in range(6)]
        
        # FPN integration using top-down pathway
        fpn_output = adapted_features.pop()  # Start with the deepest features
        for i in reversed(range(0,4)):
            upsampled = F.interpolate(fpn_output, size=adapted_features[i].shape[-2:], mode='nearest')
            fpn_output = self.fpn[f'fpn{i+1}'](upsampled + adapted_features[i])
        
        return fpn_output # 256x25x25
    
    def get_output_dims(self):
        """ Returns the number of output channels from the final convolutional layer. """
        # final_conv_layer = [layer for layer in self.fpn[-1].modules() if isinstance(layer, nn.Conv2d)][-1]
        return self.common_channel_size, 50, 50
    
    def conv_info(self):
        """
        Returns a list of dicts containing kernel sizes, strides, and paddings for each convolutional layer.
        This function will gather information from both the DenseNet backbone and the FPN custom layers.
        """
        kernel_sizes, strides, paddings = [], [], []

        # Traverse the encoder layers which are wrapped in Sequential blocks
        for seq in self.encoder:
            for layer in seq.modules():
                if isinstance(layer, nn.Conv2d):
                    kernel_sizes.append(layer.kernel_size[0])
                    strides.append(layer.stride[0])
                    paddings.append(layer.padding[0])

        # Check adaptation layers
        for layer in self.adaptation_layers.modules():
            if isinstance(layer, nn.Conv2d):
                kernel_sizes.append(layer.kernel_size[0])
                strides.append(layer.stride[0])
                paddings.append(layer.padding[0])

        # FPN layers (each one is a single Conv2d in a ModuleList)
        for layer in self.fpn.modules():
            if isinstance(layer, nn.Conv2d):
                kernel_sizes.append(layer.kernel_size[0])
                strides.append(layer.stride[0])
                paddings.append(layer.padding[0])

        return kernel_sizes, strides, paddings
    

############################################################################################################################################################################
######################################################################## Encoder-Decoder Networks ##########################################################################
############################################################################################################################################################################

class efficientDecoder_v2_s(nn.Module):
    """ EfficientNet V2 Small-based encoder-decoder architecture for feature extraction.
        Total number of parameters:  20971120 (21.00 million) 
        Returns a feature map of size 256x25x25. """ 
    def __init__(self, weights='DEFAULT', common_channel_size=None):
        """
        Initializes the efficientDecoder_v2_s class.

        Args:
            weights (str): The weights to use for the EfficientNet V2 Small features. Default is 'DEFAULT'.
            common_channel_size (int): The size of the common channel. Default is None.
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
        
        return x # 256x25x25
    
    def get_output_channels(self):
        """ Returns the number of output channels from the final convolutional layer. """
        final_conv_layer = [layer for layer in self.decoders[-1].modules() if isinstance(layer, nn.Conv2d)][-1]
        return final_conv_layer.weight.shape[1]
    
