import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import copy
import time

from src.utils.receptive_field import compute_rf_prototype
from src.utils.helpers import makedir, find_high_activation_crop

# push each prototype to the nearest patch in the training set
def push_prototypes(dataloader, # pytorch dataloader (must be unnormalized in [0,1])
                    prototype_network_parallel, # pytorch network with prototype_vectors
                    class_specific=True,
                    preprocess_input_function=None, # normalize if needed
                    prototype_layer_stride=1,
                    root_dir_for_saving_prototypes=None, # if not None, prototypes will be saved here
                    epoch_number=None, # if not provided, prototypes saved previously will be overwritten
                    prototype_img_filename_prefix=None,
                    prototype_self_act_filename_prefix=None,
                    proto_bound_boxes_filename_prefix=None,
                    save_prototype_class_identity=True, # which class the prototype image comes from
                    prototype_activation_function_in_numpy=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    prototype_network_parallel.to(device).eval()

    # Assuming prototype_network_parallel.module.prototype_vectors is a list of tensors
    all_prototype_shapes = [prototypes.shape for prototypes in prototype_network_parallel.prototype_vectors] # Num Chars x P x C x H x W
    all_n_prototypes = [shape[0] for shape in all_prototype_shapes]   # Num Chars x P
    
    # Saves the closest distance to the prototype
    all_global_min_proto_dist = [np.full(n_prototypes, np.inf) for n_prototypes in all_n_prototypes]    # Num Chars x P
    
    # Saves the patch that minimizes the distance to the prototype
    all_global_min_fmap_patches = [np.zeros([n_prototypes, shape[1], shape[2], shape[3]]) 
                                   for n_prototypes, shape in zip(all_n_prototypes, all_prototype_shapes)]  # Num Chars x P x C x H x W
    
    # Assuming the same bounding box and class identity handling applies to all characteristics
    # Initialize proto_rf_boxes and proto_bound_boxes with appropriate shapes
    if save_prototype_class_identity:
        proto_rf_boxes = [np.full([n_prototypes, 6], -1) for n_prototypes in all_n_prototypes]
        proto_bound_boxes = [np.full([n_prototypes, 6], -1) for n_prototypes in all_n_prototypes]
    else:
        proto_rf_boxes = [np.full([n_prototypes, 5], -1) for n_prototypes in all_n_prototypes]
        proto_bound_boxes = [np.full([n_prototypes, 5], -1) for n_prototypes in all_n_prototypes]

    # Create a directory to save the prototypes
    if root_dir_for_saving_prototypes != None:
        if epoch_number != None:
            proto_epoch_dir = os.path.join(root_dir_for_saving_prototypes,
                                           'epoch-'+str(epoch_number))
            makedir(proto_epoch_dir)
        else:
            proto_epoch_dir = root_dir_for_saving_prototypes
    else:
        proto_epoch_dir = None

    search_batch_size = dataloader.batch_size
    num_classes = prototype_network_parallel.num_classes

    for push_iter, (search_batch_input, search_y_chars, _, _, _, _, _) in enumerate(dataloader): # TODO: add TQDM support
        search_batch_size = search_batch_input.shape[0]
        start_index_of_search_batch = push_iter * search_batch_size

        # Handle batch processing for each characteristic
        for characteristic_index, (global_min_proto_dist, global_min_fmap_patches, proto_rf_box, proto_bound_box) in enumerate(zip(all_global_min_proto_dist, all_global_min_fmap_patches, proto_rf_boxes, proto_bound_boxes)):
            
            if root_dir_for_saving_prototypes != None:
                dir_for_saving_characteristic_prototypes = os.path.join(root_dir_for_saving_prototypes, f"characteristic_{characteristic_index}")
                os.makedirs(dir_for_saving_characteristic_prototypes, exist_ok=True)
                
            update_prototypes_on_batch(search_batch_input, 
                                       start_index_of_search_batch,
                                       prototype_network_parallel, 
                                       global_min_proto_dist,
                                       global_min_fmap_patches, 
                                       proto_rf_box, 
                                       proto_bound_box,
                                       class_specific=class_specific, 
                                       search_y=search_y_chars[characteristic_index],
                                       num_classes=prototype_network_parallel.num_classes,
                                       preprocess_input_function=preprocess_input_function,
                                       prototype_layer_stride=prototype_layer_stride,
                                       dir_for_saving_prototypes=dir_for_saving_characteristic_prototypes,
                                       prototype_img_filename_prefix=prototype_img_filename_prefix,
                                       prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                                       prototype_activation_function_in_numpy=prototype_activation_function_in_numpy,
                                       characteristic_index=characteristic_index)  # Pass characteristic_index to handle each set individually

    # Save bounding boxes and receptive field information for each set of prototypes
    if root_dir_for_saving_prototypes and epoch_number is not None:
        for idx, (proto_rf_box, proto_bound_box) in enumerate(zip(proto_rf_boxes, proto_bound_boxes)):
            np.save(os.path.join(root_dir_for_saving_prototypes, f"{proto_bound_boxes_filename_prefix}_receptive_field_characteristic_{idx}_epoch_{epoch_number}.npy"), proto_rf_box)
            np.save(os.path.join(root_dir_for_saving_prototypes, f"{proto_bound_boxes_filename_prefix}_characteristic_{idx}_epoch_{epoch_number}.npy"), proto_bound_box)

    # Update prototype vectors for each characteristic
    for idx, (prototype_vectors, global_min_fmap_patches) in enumerate(zip(prototype_network_parallel.prototype_vectors, all_global_min_fmap_patches)):
        prototype_update = np.reshape(global_min_fmap_patches, prototype_vectors.shape)
        prototype_vectors.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).to(device))


# update each prototype for current search batch
def update_prototypes_on_batch(search_batch_input,
                               start_index_of_search_batch,
                               prototype_network_parallel,
                               global_min_proto_dist, # this will be updated
                               global_min_fmap_patches, # this will be updated
                               proto_rf_boxes, # this will be updated
                               proto_bound_boxes, # this will be updated
                               class_specific=True,
                               search_y=None, # required if class_specific == True
                               num_classes=None, # required if class_specific == True
                               preprocess_input_function=None,
                               prototype_layer_stride=1,
                               dir_for_saving_prototypes=None,
                               prototype_img_filename_prefix=None,
                               prototype_self_act_filename_prefix=None,
                               prototype_activation_function_in_numpy=None,
                               characteristic_index=0):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    prototype_network_parallel.eval()

    # Preprocess the search batch
    if preprocess_input_function is not None:
        search_batch = preprocess_input_function(search_batch_input)
    else:
        search_batch = search_batch_input
        
    # Push the search batch through the network
    with torch.no_grad():
        search_batch = search_batch.to(device)
        protoL_input_torch, proto_dist_torch = prototype_network_parallel.push_forward(search_batch) # push the batch through the network
    
    # Send the data back to the cpu
    protoL_input_ = np.copy(protoL_input_torch.detach().cpu().numpy())  # (batch_size, C, H, W)
    proto_dist_ = np.copy(proto_dist_torch[characteristic_index].detach().cpu().numpy())    # (batch_size, num_prototypes_per_characteristic, H, W)

    del protoL_input_torch, proto_dist_torch

    # Initialize class_to_img_index_dict
    if class_specific:
        class_to_img_index_dict = {key: [] for key in range(num_classes)}
        for img_index, img_y in enumerate(search_y):
            img_label = img_y.item()
            class_to_img_index_dict[img_label].append(img_index)
            
    prototype_shape = prototype_network_parallel.prototype_shape
    n_prototypes = prototype_network_parallel.prototypes_per_characteristic
    proto_h = prototype_shape[2]
    proto_w = prototype_shape[3]
    max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3] # C * H * W

    for j in range(n_prototypes):
        if class_specific:
            # target_class is the class of the class_specific prototype
            target_class = torch.argmax(prototype_network_parallel.prototype_class_identity[characteristic_index][j]).item()
            # if there is not images of the target_class from this batch
            # we go on to the next prototype
            if len(class_to_img_index_dict[target_class]) == 0:
                continue
            proto_dist_j = proto_dist_[class_to_img_index_dict[target_class]][:,j,:,:]
        else:
            # if it is not class specific, then we will search through every example
            proto_dist_j = proto_dist_[:,j,:,:]

        batch_min_proto_dist_j = np.amin(proto_dist_j)
        
        if batch_min_proto_dist_j < global_min_proto_dist[j]:
            batch_argmin_proto_dist_j = list(np.unravel_index(np.argmin(proto_dist_j, axis=None), proto_dist_j.shape)) # [feature_index, h, w]
            
            if class_specific:
                '''
                change the argmin index from the index among
                images of the target class to the index in the entire search
                batch
                '''
                batch_argmin_proto_dist_j[0] = class_to_img_index_dict[target_class][batch_argmin_proto_dist_j[0]]

            # retrieve the corresponding feature map patch
            img_index_in_batch = batch_argmin_proto_dist_j[0]
            fmap_height_start_index = batch_argmin_proto_dist_j[1] * prototype_layer_stride
            fmap_height_end_index = fmap_height_start_index + proto_h
            fmap_width_start_index = batch_argmin_proto_dist_j[2] * prototype_layer_stride
            fmap_width_end_index = fmap_width_start_index + proto_w

            batch_min_fmap_patch_j = protoL_input_[img_index_in_batch,
                                                   :,
                                                   fmap_height_start_index:fmap_height_end_index,
                                                   fmap_width_start_index:fmap_width_end_index] # Batch x C x H x W

            global_min_proto_dist[j] = batch_min_proto_dist_j
            global_min_fmap_patches[j] = batch_min_fmap_patch_j
            
            # get the receptive field boundary of the image patch
            # that generates the representation
            protoL_rf_info = prototype_network_parallel.proto_layer_rf_info
            rf_prototype_j = compute_rf_prototype(search_batch.size(2), batch_argmin_proto_dist_j, protoL_rf_info)
            
            # get the whole image
            original_img_j = search_batch_input[rf_prototype_j[0]]
            original_img_j = original_img_j.numpy()
            original_img_j = np.transpose(original_img_j, (1, 2, 0))
            original_img_size = original_img_j.shape[0]
            
            # crop out the receptive field
            rf_img_j = original_img_j[rf_prototype_j[1]:rf_prototype_j[2],
                                      rf_prototype_j[3]:rf_prototype_j[4], :]
            
            # save the prototype receptive field information
            proto_rf_boxes[j, 0] = rf_prototype_j[0] + start_index_of_search_batch
            proto_rf_boxes[j, 1] = rf_prototype_j[1]
            proto_rf_boxes[j, 2] = rf_prototype_j[2]
            proto_rf_boxes[j, 3] = rf_prototype_j[3]
            proto_rf_boxes[j, 4] = rf_prototype_j[4]
            if proto_rf_boxes.shape[1] == 6 and search_y is not None:
                proto_rf_boxes[j, 5] = search_y[rf_prototype_j[0]].item()

            # find the highly activated region of the original image
            proto_dist_img_j = proto_dist_[img_index_in_batch, j, :, :]
            if prototype_network_parallel.prototype_activation_function == 'log':
                proto_act_img_j = np.log((proto_dist_img_j + 1) / (proto_dist_img_j + prototype_network_parallel.epsilon))
            elif prototype_network_parallel.prototype_activation_function == 'linear':
                proto_act_img_j = max_dist - proto_dist_img_j
            else:
                proto_act_img_j = prototype_activation_function_in_numpy(proto_dist_img_j)
            upsampled_act_img_j = cv2.resize(proto_act_img_j, dsize=(original_img_size, original_img_size),
                                             interpolation=cv2.INTER_CUBIC)
            proto_bound_j = find_high_activation_crop(upsampled_act_img_j)
            # crop out the image patch with high activation as prototype image
            proto_img_j = original_img_j[proto_bound_j[0]:proto_bound_j[1],
                                         proto_bound_j[2]:proto_bound_j[3], :]

            # save the prototype boundary (rectangular boundary of highly activated region)
            proto_bound_boxes[j, 0] = proto_rf_boxes[j, 0]
            proto_bound_boxes[j, 1] = proto_bound_j[0]
            proto_bound_boxes[j, 2] = proto_bound_j[1]
            proto_bound_boxes[j, 3] = proto_bound_j[2]
            proto_bound_boxes[j, 4] = proto_bound_j[3]
            
            if proto_bound_boxes.shape[1] == 6 and search_y is not None:
                proto_bound_boxes[j, 5] = search_y[rf_prototype_j[0]].item()

            if dir_for_saving_prototypes is not None:
                if prototype_self_act_filename_prefix is not None:
                    # save the numpy array of the prototype self activation
                    np.save(os.path.join(dir_for_saving_prototypes,
                                         prototype_self_act_filename_prefix + str(j) + '.npy'),
                            proto_act_img_j)
                if prototype_img_filename_prefix is not None:
                    # save the whole image containing the prototype as png
                    plt.imsave(os.path.join(dir_for_saving_prototypes,
                                            prototype_img_filename_prefix + '-original' + str(j) + '.png'),
                               original_img_j,
                               vmin=0.0,
                               vmax=1.0)
                    # overlay (upsampled) self activation on original image and save the result
                    rescaled_act_img_j = upsampled_act_img_j - np.amin(upsampled_act_img_j)
                    rescaled_act_img_j = rescaled_act_img_j / np.amax(rescaled_act_img_j)
                    heatmap = cv2.applyColorMap(np.uint8(255*rescaled_act_img_j), cv2.COLORMAP_JET)
                    heatmap = np.float32(heatmap) / 255
                    heatmap = heatmap[...,::-1]
                    overlayed_original_img_j = 0.5 * original_img_j + 0.3 * heatmap
                    plt.imsave(os.path.join(dir_for_saving_prototypes,
                                            prototype_img_filename_prefix + '-original_with_self_act' + str(j) + '.png'),
                               overlayed_original_img_j,
                               vmin=0.0,
                               vmax=1.0)
                    
                    # if different from the original (whole) image, save the prototype receptive field as png
                    if rf_img_j.shape[0] != original_img_size or rf_img_j.shape[1] != original_img_size:
                        plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                prototype_img_filename_prefix + '-receptive_field' + str(j) + '.png'),
                                   rf_img_j,
                                   vmin=0.0,
                                   vmax=1.0)
                        overlayed_rf_img_j = overlayed_original_img_j[rf_prototype_j[1]:rf_prototype_j[2],
                                                                      rf_prototype_j[3]:rf_prototype_j[4]]
                        plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                prototype_img_filename_prefix + '-receptive_field_with_self_act' + str(j) + '.png'),
                                   overlayed_rf_img_j,
                                   vmin=0.0,
                                   vmax=1.0)
                    
                    # save the prototype image (highly activated region of the whole image)
                    plt.imsave(os.path.join(dir_for_saving_prototypes,
                                            prototype_img_filename_prefix + str(j) + '.png'),
                               proto_img_j,
                               vmin=0.0,
                               vmax=1.0)
                
    if class_specific:
        del class_to_img_index_dict