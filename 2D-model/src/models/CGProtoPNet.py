import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from src.utils.receptive_field import compute_proto_layer_rf_info_v2
from sklearn.decomposition import PCA
import numpy as np
import save
from settings import base_architecture, experiment_run

from resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet101_features, resnet152_features
from densenet_features import densenet121_features, densenet161_features, densenet169_features, densenet201_features
from vgg_features import vgg11_features, vgg11_bn_features, vgg13_features, vgg13_bn_features, vgg16_features, vgg16_bn_features,\
                        vgg19_features, vgg19_bn_features

base_architecture_to_features = {'resnet18': resnet18_features,
                                 'resnet34': resnet34_features,
                                 'resnet50': resnet50_features,
                                 'resnet101': resnet101_features,
                                 'resnet152': resnet152_features,
                                 'densenet121': densenet121_features,
                                 'densenet161': densenet161_features,
                                 'densenet169': densenet169_features,
                                 'densenet201': densenet201_features,
                                 'vgg11': vgg11_features,
                                 'vgg11_bn': vgg11_bn_features,
                                 'vgg13': vgg13_features,
                                 'vgg13_bn': vgg13_bn_features,
                                 'vgg16': vgg16_features,
                                 'vgg16_bn': vgg16_bn_features,
                                 'vgg19': vgg19_features,
                                 'vgg19_bn': vgg19_bn_features}

class PPNet(nn.Module):

    def __init__(self, features, img_size, prototype_shape,
                 proto_layer_rf_info, num_classes, num_characteristics, init_weights=True,
                 prototype_activation_function='log', add_on_layers_type='bottleneck'):
        """
        Initializes the PPNet class.

        Args:
            features (nn.Module): The feature extraction module.
            img_size (int): The size of the input image.
            prototype_shape (tuple): The shape of the prototypes.
            proto_layer_rf_info (list): The receptive field information for each prototype layer.
            num_classes (int): The number of classes.
            num_characteristics (int): The number of characteristics.
            init_weights (bool, optional): Whether to initialize the weights. Defaults to True.
            prototype_activation_function (str, optional): The activation function for the prototypes. Defaults to 'log'.
            add_on_layers_type (str, optional): The type of additional layers to add. Defaults to 'bottleneck'.
        """

        super(PPNet, self).__init__()
        self.img_size = img_size
        self.prototype_shape = prototype_shape
        self.num_prototypes = prototype_shape[0]
        self.dim_prototype = prototype_shape[1]
        print("self.dim_prototype: "+str(self.dim_prototype))
        self.num_classes = num_classes
        self.num_characteristics = num_characteristics
        self.epsilon = 1e-4
        self.count = 50
        self.extra_image_count = 0
        
        # prototype_activation_function could be 'log', 'linear',
        # or a generic function that converts distance to similarity score
        self.prototype_activation_function = prototype_activation_function
        '''
        Here we are initializing the class identities of the prototypes
        Without domain specific knowledge we allocate the same number of
        prototypes for each class
        '''
        assert(self.num_prototypes % self.num_classes == 0)
        # a onehot indication matrix for each prototype's class identity
        #dm I now consider this as the class for a certain characteristic
        
        self.prototype_class_identity = torch.zeros(self.num_prototypes,
                                                    self.num_classes)
        
        
        #dm I create a onehot matrix to hold the characteristic that each prototype represents
        self.prototype_characteristic_identity = torch.zeros(self.num_prototypes,
                                                    self.num_characteristics)
        
        #dm modified to assign prototypes to classes of characteristics
        self.prototype_identities = [] #list with tuple entry for each prototype describing its identity as (class, characteristic)
        self.prototype_characteristics = [] #redundant as information exists in prototype identities but simpler to use

        # Assign class and characteristic identities to each prototype
        for j in range(self.num_prototypes):
            char_identity = j // (self.num_prototypes // self.num_characteristics)
            class_identity = char_identity % self.num_classes # Distribute prototypes evenly among classes
            self.prototype_characteristic_identity[j, char_identity] = 1 
            self.prototype_class_identity[j, class_identity] = 1 
            self.prototype_identities.append((class_identity, char_identity))
            self.prototype_characteristics.append(char_identity)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.prototype_class_identity = self.prototype_class_identity.to(self.device)

        #Create a list of masks for each characteristic
        #where each mask can be multiplied with prototype_class_identity to only select prototypes for a specific characteristic
        self.characteristic_masks = []
        for char_index in range(self.num_characteristics):
            rows = []
            for prototp in self.prototype_identities:
                prototype_characteristic_identity = prototp[1]
                if prototype_characteristic_identity == char_index:
                    row = torch.ones(num_classes)
                else:
                    row = torch.zeros(num_classes)
                rows.append(row)
            self.characteristic_masks.append(torch.stack(rows))

        #list of indices of prototypes that belong to each characteristic
        self.characteristic_indices = {}
        for prototype_index, characteristic in enumerate(self.prototype_characteristics):
            if characteristic not in self.characteristic_indices:
                self.characteristic_indices[characteristic] = [prototype_index]
            else:
                self.characteristic_indices[characteristic].append(prototype_index)

        self.num_output_nodes = self.num_classes * self.num_characteristics
        self.positive_one_weights_locations = torch.zeros(self.num_prototypes, self.num_output_nodes)
        self.negative_one_weights_locations = torch.zeros(self.num_prototypes, self.num_output_nodes)

        for prototype_index in range(self.num_prototypes):
            #set the connections between a prototype and its corresponding output node from this characteristic
            self.positive_one_weights_locations[prototype_index, self.prototype_identities[prototype_index][0] + self.prototype_identities[prototype_index][1]*self.num_classes] = 1
            #set the connections between a prototype and the other classes of this characteristic to 1 (will be multiplied by negative factor later)
            for class_index in range(self.num_classes):
                if class_index != self.prototype_identities[prototype_index][0]:
                    self.negative_one_weights_locations[prototype_index, class_index + self.prototype_identities[prototype_index][1]*self.num_classes] = 1

        self.proto_layer_rf_info = proto_layer_rf_info

        self.features = features

        features_name = str(self.features).upper()
        if features_name.startswith('VGG') or features_name.startswith('RES'):
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
        elif features_name.startswith('DENSE'):
            # If the architecture name starts with 'DENSE', indicating a DenseNet model, the code instead looks for the last batch 
            # normalization layer (nn.BatchNorm2d) in the features module and retrieves its number of features (num_features). 
            # This value is then assigned to the first_add_on_layer_in_channels variable
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.BatchNorm2d)][-1].num_features
        else:
            raise Exception('other base base_architecture NOT implemented')

        #My inverse feature pyramid layers
        #Feature Pyramid Convolution (first two layers)
        self.conv_pyramid_1_2 = nn.Conv2d(in_channels=(128+256), out_channels=256, kernel_size=1)
        #Feature Pyramid Convolution ((first two combination) and third layers)
        self.conv_pyramid_1_2_3 = nn.Conv2d(in_channels=(256+1024), out_channels=256, kernel_size=1)

        # #Feature pyramid layers
        # #Feature Pyramid Convolution (first two layers)
        # self.conv_pyramid_3_2 = nn.Conv2d(in_channels=(1024+256), out_channels=256, kernel_size=1)
        # #Feature Pyramid Convolution ((first two combination) and third layers)
        # self.conv_pyramid_3_2_1 = nn.Conv2d(in_channels=(128+256), out_channels=256, kernel_size=1)

        # Feature pyramid layers
        #Feature Pyramid Convolution (first two layers)
        self.conv_pyramid_3_2 = nn.Conv2d(in_channels=(1024+256), out_channels=128, kernel_size=1)
        #Feature Pyramid Convolution ((first two combination) and third layers)
        self.conv_pyramid_3_2_1 = nn.Conv2d(in_channels=(128+128), out_channels=128, kernel_size=1)

        #Faithful Feature Pyramid Layers
        #individual 1x1 convolutions for each intermediate feature map
        self.conv_pyramid_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1)
        self.conv_pyramid_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1)
        self.conv_pyramid_3 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)



        #overwrite in channels for the concatenated version
        first_add_on_layer_in_channels = 128

        if add_on_layers_type == 'bottleneck':
            add_on_layers = []
            current_in_channels = first_add_on_layer_in_channels
            #add batchnorm and relu that I removed from densenet
            add_on_layers.append(nn.BatchNorm2d(current_in_channels))
            add_on_layers.append(nn.ReLU())
            while (current_in_channels > self.prototype_shape[1]) or (len(add_on_layers) == 0):
                current_out_channels = max(self.prototype_shape[1], (current_in_channels // 2))
                add_on_layers.append(nn.Conv2d(in_channels=current_in_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                add_on_layers.append(nn.ReLU())
                add_on_layers.append(nn.Conv2d(in_channels=current_out_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                if current_out_channels > self.prototype_shape[1]:
                    add_on_layers.append(nn.ReLU())
                else:
                    assert(current_out_channels == self.prototype_shape[1])
                    add_on_layers.append(nn.Sigmoid())
                current_in_channels = current_in_channels // 2
            self.add_on_layers = nn.Sequential(*add_on_layers)
        else:
            self.add_on_layers = nn.Sequential(
                nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=self.prototype_shape[1], kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=self.prototype_shape[1], out_channels=self.prototype_shape[1], kernel_size=1),
                nn.Sigmoid()
            )
        
        ####################################################################################################################
        ###################################### Initialize the prototype vectors ############################################
        ####################################################################################################################
        
        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape),
                                              requires_grad=True)
        
        self.ones = nn.Parameter(torch.ones(self.prototype_shape),
                                              requires_grad=False)  

        ####################################################################################################################
        ####################################################################################################################
        

        self.last_layer = nn.Linear(self.num_prototypes, self.num_output_nodes, bias=False) # do not use bias
        self.final_activation = nn.Softmax()#nn.Sigmoid() #removed the sigmoid activation as it was only needed for the binary case of model_4 (now model_4_5 is multi-class)

        ######################################### Create the prototype class layer (for ordering) #########################################
        
        self.prototype_class_layer_num_output_nodes = self.num_classes*self.num_characteristics
        self.prototype_class_layer = nn.Linear(self.dim_prototype, self.prototype_class_layer_num_output_nodes)

        self.affinity_matrix = torch.zeros(self.num_prototypes, self.num_prototypes)
        for i in range(self.num_prototypes):
            for j in range(self.num_prototypes):
                class_i = self.prototype_identities[i][0]
                class_j = self.prototype_identities[j][0]

                if class_i == class_j:
                    self.affinity_matrix[i,j] = 1
                elif abs(class_i - class_j) == 1:
                    self.affinity_matrix[i,j] = 0.5
        
        #affinity matrix for multiple characteristics
        self.affinity_matrix_char = torch.zeros(self.num_prototypes, self.num_prototypes)
        for i in range(self.num_prototypes):
            for j in range(self.num_prototypes):
                class_i = self.prototype_identities[i][0]
                class_j = self.prototype_identities[j][0]

                if (class_i == class_j) and (self.prototype_identities[i][1] == self.prototype_identities[j][1]):
                    self.affinity_matrix_char[i,j] = 1
                elif (abs(class_i - class_j) == 1) and (self.prototype_identities[i][1] == self.prototype_identities[j][1]):
                    self.affinity_matrix_char[i,j] = 0.5
        
        self.my_laplacian_matrix = torch.zeros(self.num_prototypes, self.num_prototypes)
        self.my_laplacian_matrix_char = torch.zeros(self.num_prototypes, self.num_prototypes)
        affinity_matrix_row_sums = torch.sum(self.affinity_matrix, dim=1)
        affinity_matrix_row_sums_char = torch.sum(self.affinity_matrix_char, dim=1)

        prototypes_per_class = self.num_prototypes // self.num_classes
        prototypes_per_class_char = self.num_prototypes // (self.num_classes*self.num_characteristics)
        
        #my initial single characteristic laplacian matrix
        for i in range(self.num_prototypes):
            row_sum = affinity_matrix_row_sums[i]
            for j in range(self.num_prototypes):
                class_i = self.prototype_identities[i][0]
                class_j = self.prototype_identities[j][0]

                if class_i == class_j:
                    self.my_laplacian_matrix[i,j] = row_sum/prototypes_per_class - 1
                elif abs(class_i - class_j) == 1:
                    self.my_laplacian_matrix[i,j] = -self.affinity_matrix[i,j]

        #my new multi characteristic laplacian matrix
        for i in range(self.num_prototypes):
            row_sum = affinity_matrix_row_sums_char[i]
            for j in range(self.num_prototypes):
                class_i = self.prototype_identities[i][0]
                class_j = self.prototype_identities[j][0]

                if (class_i == class_j) and (self.prototype_identities[i][1] == self.prototype_identities[j][1]):
                    self.my_laplacian_matrix_char[i,j] = row_sum/prototypes_per_class_char - 1
                elif (abs(class_i - class_j) == 1) and (self.prototype_identities[i][1] == self.prototype_identities[j][1]):
                    self.my_laplacian_matrix_char[i,j] = -self.affinity_matrix_char[i,j]
        
        self.my_laplacian_matrix_char = self.my_laplacian_matrix_char.to(self.device)

                
        
        self.degree_matrix = torch.diag(torch.sum(self.affinity_matrix, dim=1))

        self.laplacian_matrix = self.degree_matrix - self.affinity_matrix

        #regression
        self.pca_char = []
        self.regression_scalar_char = []
        for char_index in range(self.num_characteristics):
            pca = PCA(n_components=1)
            self.pca_char.append(pca)
            regression_scalar = nn.Parameter(torch.rand(1), requires_grad=True)
            self.regression_scalar_char.append(regression_scalar)
        


        #characteristic mask index for selecting the output for the relevant characteristic for the pototype classification layer output
        self.prot_class_layer_characteristic_mask_indices = []
        for char_index in range(self.num_characteristics):
            characteristic_mask_index = torch.tensor([i for i in range(self.prototype_class_layer_num_output_nodes) if i // (num_classes) ==  char_index])
            self.prot_class_layer_characteristic_mask_indices.append(characteristic_mask_index)
        
        #relevant prototype classification layer output indices for each characteristic
        self.prot_class_layer_characteristic_output_indices = [self.prot_class_layer_characteristic_mask_indices[prototype_char_index] for prototype_char_index in self.prototype_characteristics]
        print("prot_class_layer_characteristic_output_indices: "+str(self.prot_class_layer_characteristic_output_indices))

        if init_weights:
            self._initialize_weights()

        ######################################### Create the dense layer (that aids with malignancy prediction) #########################################
        #100*46*46*10
        self.dense_layer = nn.Linear(int(14400), self.num_characteristics*self.num_classes)
        ######################################### Create the final prediction layer (for malignancy prediction) #########################################
        penultimate_layer_output_params = round(self.num_characteristics*self.num_classes/2)
        self.penultimate_prediction_layer = nn.Linear(self.num_characteristics*self.num_classes, penultimate_layer_output_params)
        self.final_prediction_layer = nn.Linear(penultimate_layer_output_params, 1) #the layer that takes the characteristic output and returns the final malignancy prediction
        

    def conv_features(self, x):
        '''
        the feature input to prototype layer
        '''
        x, intermediate_outputs = self.features(x)

        relevant_intermediate_outputs = intermediate_outputs[:-1]
        
        #apply batch normalization and relu to each output
        normalized_intermediate_outputs = []
        relu = nn.ReLU().to(self.device)
        for intermediate_output in relevant_intermediate_outputs:
            bn = nn.BatchNorm2d(num_features=intermediate_output.shape[1]).to(intermediate_output.device)
            normalized_intermediate_outputs.append(relu(bn(intermediate_output)))

        normalized_intermediate_outputs.append(x)

        ########################################################################################
        #############        Feature Pyramid implementation       ##############################
        ########################################################################################

        #upsample the smallest feature map to the size of the second
        upsampled_3 = F.interpolate(normalized_intermediate_outputs[2], size=(6, 6), mode='bicubic')
        #concatenate the upsampled_1 with the second feature map
        concatenated_3_2 = torch.cat([normalized_intermediate_outputs[1], upsampled_3], dim=1)
        #apply 1x1 convolution to reduce the channel dimensions and merge (+Relu)
        merged_3_2 = self.conv_pyramid_3_2(concatenated_3_2)        

        upsampled_3_2 = F.interpolate(merged_3_2, size=(12, 12), mode='bicubic')

        concatenated_3_2_1 = torch.cat([normalized_intermediate_outputs[0], upsampled_3_2], dim=1)

        merged_3_2_1 = self.conv_pyramid_3_2_1(concatenated_3_2_1)
    
        merged_3_2_1 = self.add_on_layers(merged_3_2_1)
        
        ########################################################################################

        return merged_3_2_1

    def _l2_convolution(self, x):
        '''
        apply self.prototype_vectors as l2-convolution filters on input x
        '''
        x2 = x ** 2
        x2_patch_sum = F.conv2d(input=x2, weight=self.ones)

        p2 = self.prototype_vectors ** 2
        p2 = torch.sum(p2, dim=(1, 2, 3))
        p2_reshape = p2.view(-1, 1, 1)

        xp = F.conv2d(input=x, weight=self.prototype_vectors)
        intermediate_result = - 2 * xp + p2_reshape  # use broadcast
        distances = F.relu(x2_patch_sum + intermediate_result) #dm sum of squares depthwise + simple sum depthwise + 2 x (prototype activation map for each prototype)
        #dm thus the distance: 
        # - is smaller when a prototypes has higher activation on a patch
        # - is normalised by summing the patch depthwise as that would artificially increase the activation without the tiles actually matching
        # - is normalised by summing the squared values of the patch depthwise as above
        return distances

    def prototype_classification(self):
        
        #Yah: This currently outputs two sets of class probabilities, one for each characteristic
        # I thus have to modify this such that each output (corresponding to a prototype) only contains number_of_classes outputs.
        # These should be the ones that correspond to that characteristic.
        #Output probabilities are organised such that the first num_classes correspond to the first characteristic, the second num_classes to the second characteristic and so on
        #The full prototype class predictions are first calculated
        #Then for each entry, the correct num_classes outputs are selected
        #Softmax is then applied as normal
        prototype_class_predictions = self.prototype_class_layer(self.prototype_vectors.view(self.num_prototypes, -1))
        # print("prototype_class_predictions: ")
        # print(prototype_class_predictions)

        prototype_class_predictions_char = torch.stack([prototype_class_predictions[prototype_index, self.prot_class_layer_characteristic_output_indices[prototype_index]] for prototype_index in range(self.num_prototypes)])
        # print("prototype_class_predictions_char: "+str(prototype_class_predictions_char))
        prototype_class_probabilities = F.softmax(prototype_class_predictions_char, dim=1)

        return prototype_class_probabilities

    def prototype_distances(self, x):
        '''
        x is the raw input
        '''
        conv_features = self.conv_features(x) # backbone feature extraction
        distances = self._l2_convolution(conv_features) # prototype distance calculation
        return distances, conv_features

    def distance_2_similarity(self, distances):
        if self.prototype_activation_function == 'log':
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif self.prototype_activation_function == 'linear':
            return -distances
        else:
            return self.prototype_activation_function(distances)

    def prototype_pca(self):

        #perform pca for each characteristic
        for char_index in range(self.num_characteristics):
            #get the protototypes for this characteristic
            prototypes_non_squeezed = self.prototype_vectors[self.characteristic_indices[char_index],:,:,:].cpu().detach().numpy()
            prototypes = np.squeeze(prototypes_non_squeezed)
            print("prototypes shape: "+str(prototypes.shape))

            #perform pca on the prototype vectors (1D embedding)
            self.pca_char[char_index].fit(prototypes)


    def prototype_regression(self, x):
        
        #refresh prototype pca
        self.prototype_pca()

        #get the embedding of the input
        conv_features = self.conv_features(x)

        #for each regressive characteristic calculate the projection of the closest patch to the line on the regression line
        closetst_reg_values_char = []
        for char_index in range(self.num_characteristics):
            char_pca = self.pca_char[char_index]

            #find the closest patch in the input to the regression line of this characteristic
            #loop through each patch in the input
            print("conv_features shape: "+str(conv_features.shape))
            #unfold the convolutional feature map into patches
            conv_features_unf = F.unfold(conv_features, kernel_size=(1,1))

            #unfold elements of batch (get list of all patches for all images in batch)
            #this ensures that distance calculation is done separately for each patch
            conv_features_unf = conv_features_unf.view(-1, self.dim_prototype)
            
            patch_distances, patch_projections = self.patches_to_line_dist(conv_features_unf, char_pca)

            print("min patch distance: "+str(torch.min(patch_distances)))
            
            
            #reshape patch distances to batch
            patch_distances = patch_distances.reshape(conv_features.shape[0], conv_features.shape[2]*conv_features.shape[3])
            
            #get the flattened out index for the minimum distance patch in each image of the batch
            print("patch_distances: "+str(patch_distances))
            print("patch_distances shape: "+str(patch_distances.shape))

            #get the flattened out index for the minimum distance patch in each image of the batch
            min_distance_index = torch.argmin(patch_distances, axis=1)


            min_distance = torch.min(patch_distances, axis=1)
            print("min_distance_index: "+str(min_distance_index))
            print("min_distance_index shape: "+str(min_distance_index.shape))
            print("min_distance: "+str(min_distance))

            self.count -= 1
            if self.count <= -10000:
                #create a corresponding index for each patch in the batch
                batch_min_distance_indices = []
                row = 0
                for i in min_distance_index:
                    batch_min_distance_indices.append(i+conv_features.shape[2]*conv_features.shape[3]*row)
                    row += 1
                batch_min_distance_indices = torch.tensor(batch_min_distance_indices)

                model_dir = './saved_models/' + base_architecture + '/' + experiment_run + '/'
                save.save_pca(self, model_dir, 'pca', 0, extra_points=patch_projections.detach().cpu().numpy())
                save.save_prototype_latent_space(self, model_dir, 'latent_'+str(self.extra_image_count), 0, extra_points=conv_features_unf.detach().cpu().numpy(), extra_points_special_indices=batch_min_distance_indices.detach().cpu().numpy(), line=char_pca.components_.squeeze())
                save.save_prototype_latent_spaces(self, model_dir, 'latent_'+str(self.extra_image_count), 0, extra_points=conv_features_unf.detach().cpu().numpy(), extra_points_special_indices=batch_min_distance_indices.detach().cpu().numpy(), line=char_pca.components_.squeeze())
                self.extra_image_count += 1
                self.count = 50

            #make into same dimension as patch distances
            min_distance_index_fer = min_distance_index.unsqueeze(1)

            patch_reg_values = patch_projections.reshape(conv_features.shape[0], conv_features.shape[2]*conv_features.shape[3])

            closest_reg_values = patch_reg_values.gather(1, min_distance_index_fer)
            print("closest_reg_values: "+str(closest_reg_values))

            print("min patch projection: "+str(torch.min(patch_projections)))
            print("max patch projection: "+str(torch.max(patch_projections)))
            print("max closest_reg_values: "+str(torch.max(closest_reg_values)))
            print("min closest_reg_values: "+str(torch.min(closest_reg_values)))
            print("mean closest_reg_values: "+str(torch.mean(closest_reg_values)))
            

            closetst_reg_values_char.append(closest_reg_values)
        
        return closetst_reg_values_char


    def patches_to_line_dist(self, patches, pca):
        
        line = pca.components_.squeeze()
        line = torch.tensor(line).to(self.device)

        #centering the patches 
        #This is done as the sklearn pca automatically centers the data for fit
        patches_centered = patches - torch.tensor(pca.mean_).to(self.device)

        #project each patch on the line
        projections_in_line_space = torch.matmul(patches_centered, line)
        projections = projections_in_line_space.reshape(-1,1) * line

        #calculate the vector connecting each patch and its projection
        projection_to_patch_line = patches_centered - projections
        
        #calculate the patch to line ditances
        patch_to_line_distances = torch.norm(projection_to_patch_line, dim=1)
        
        return patch_to_line_distances, projections_in_line_space

    def forward(self, x):
        distances, _ = self.prototype_distances(x)

        # global min pooling
        min_distances = -F.max_pool2d(-distances,
                                      kernel_size=(distances.size()[2],
                                                   distances.size()[3]))

        min_distances = min_distances.view(-1, self.num_prototypes)
        
        prototype_activations = self.distance_2_similarity(min_distances)
        
        logits = self.last_layer(prototype_activations) 
        
        prototype_class_probabilities = self.prototype_classification() # prototype classification probabilities independent of input

        reg_values_char = self.prototype_regression(x)  # regression values for each characteristic

        distances_f = distances.view(distances.shape[0], -1) # flatten the distances
        
        #apply a dense layer to the flattened distances
        distances_f = self.dense_layer(distances_f)
        penultimate_layer_output = self.penultimate_prediction_layer(distances_f)
        penultimate_layer_output = F.relu(penultimate_layer_output)
        final_prediction_logit = self.final_prediction_layer(penultimate_layer_output)
        final_prediction = torch.sigmoid(final_prediction_logit)    # Final Malignancy Prediction

        return logits, min_distances, prototype_class_probabilities, reg_values_char, final_prediction

    def push_forward(self, x):
        '''this method is needed for the pushing operation'''
        conv_output = self.conv_features(x)
        distances = self._l2_convolution(conv_output)
        return conv_output, distances

    def prune_prototypes(self, prototypes_to_prune):
        '''
        prototypes_to_prune: a list of indices each in
        [0, current number of prototypes - 1] that indicates the prototypes to
        be removed
        '''
        prototypes_to_keep = list(set(range(self.num_prototypes)) - set(prototypes_to_prune))

        self.prototype_vectors = nn.Parameter(self.prototype_vectors.data[prototypes_to_keep, ...],
                                              requires_grad=True)

        self.prototype_shape = list(self.prototype_vectors.size())
        self.num_prototypes = self.prototype_shape[0]

        # changing self.last_layer in place
        # changing in_features and out_features make sure the numbers are consistent
        self.last_layer.in_features = self.num_prototypes
        self.last_layer.out_features = self.num_output_nodes
        self.last_layer.weight.data = self.last_layer.weight.data[:, prototypes_to_keep]

        # self.ones is nn.Parameter
        self.ones = nn.Parameter(self.ones.data[prototypes_to_keep, ...],
                                 requires_grad=False)
        # self.prototype_class_identity is torch tensor
        # so it does not need .data access for value update
        self.prototype_class_identity = self.prototype_class_identity[prototypes_to_keep, :]

    def __repr__(self):
        # PPNet(self, features, img_size, prototype_shape,
        # proto_layer_rf_info, num_classes, init_weights=True):
        rep = (
            'PPNet(\n'
            '\tfeatures: {},\n'
            '\timg_size: {},\n'
            '\tprototype_shape: {},\n'
            '\tproto_layer_rf_info: {},\n'
            '\tnum_classes: {},\n'
            '\tepsilon: {}\n'
            ')'
        )

        return rep.format(self.features,
                          self.img_size,
                          self.prototype_shape,
                          self.proto_layer_rf_info,
                          self.num_classes,
                          self.epsilon)

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        '''
        the incorrect strength will be actual strength if -0.5 then input -0.5
        '''
        
        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        a = correct_class_connection * self.positive_one_weights_locations + incorrect_class_connection * self.negative_one_weights_locations
        
        print("self.last_layer.weight.data shape:", self.last_layer.weight.data.shape)
        print(a.shape)
        self.last_layer.weight.data.copy_(
            (correct_class_connection * self.positive_one_weights_locations
            + incorrect_class_connection * self.negative_one_weights_locations).T)

    def _initialize_weights(self):
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        #self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)
        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)



def construct_PPNet(base_architecture, pretrained=True, img_size=224,
                    prototype_shape=(2000, 512, 1, 1), num_classes=5, num_characteristics=1,
                    prototype_activation_function='lo', add_on_layers_type='bottleneck'):

    features = base_architecture_to_features[base_architecture](pretrained=pretrained)
    
    layer_filter_sizes, layer_strides, layer_paddings = features.conv_info()#([3,3],[1,1],[1,1])
    proto_layer_rf_info = compute_proto_layer_rf_info_v2(img_size=img_size,
                                                         layer_filter_sizes=layer_filter_sizes,
                                                         layer_strides=layer_strides,
                                                         layer_paddings=layer_paddings,
                                                         prototype_kernel_size=prototype_shape[2])
    return PPNet(features=features,
                 img_size=img_size,
                 prototype_shape=prototype_shape,
                 proto_layer_rf_info=proto_layer_rf_info,
                 num_classes=num_classes,
                 init_weights=True,
                 prototype_activation_function=prototype_activation_function,
                 add_on_layers_type=add_on_layers_type, num_characteristics=num_characteristics)

