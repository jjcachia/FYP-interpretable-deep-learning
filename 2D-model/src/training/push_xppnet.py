import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import time
from tqdm import tqdm

def push_prototypes(
    dataloader,  # pytorch dataloader
    # dataset,   # pytorch dataset for train_push group
    # prototype_layer_stride=1,
    model,  # pytorch network with feature encoder and prototype vectors
    class_specific=True,  # enable pushing protos from only the alotted class
    abstain_class=True,  # indicates K+1-th class is of the "abstain" type
    preprocess_input_function=None,  # normalize if needed
    root_dir_for_saving_prototypes=None,  # if not None, prototypes will be saved in this dir
    epoch_number=None,  # if not provided, prototypes saved previously will be overwritten
    log=print,
    prototype_img_filename_prefix=None,
    prototype_self_act_filename_prefix=None,
    proto_bound_boxes_filename_prefix=None,
    replace_prototypes=True,
):
    """
    Search the training set for image patches that are semantically closest to
    each learned prototype, then updates the prototypes to those image patches.

    To do this, it computes the image patch embeddings (IPBs) and saves those
    closest to the prototypes. It also saves the prototype-to-IPB distances and
    predicted occurrence maps.

    If abstain_class==True, it assumes num_classes actually equals to K+1, where
    K is the number of real classes and 1 is the extra "abstain" class for
    uncertainty estimation.
    """

    model.eval()
    log(f"############## push at epoch {epoch_number} #################")

    start = time.time()

    # creating the folder (with epoch number) to save the prototypes' info and visualizations
    if root_dir_for_saving_prototypes != None:
        if epoch_number != None:
            proto_epoch_dir = os.path.join(root_dir_for_saving_prototypes, "epoch-" + str(epoch_number))
            makedir(proto_epoch_dir)
        else:
            proto_epoch_dir = root_dir_for_saving_prototypes
    else:
        proto_epoch_dir = None

    # find the number of prototypes, and number of classes for this push
    prototype_shape = model.prototype_shape  # shape (P, D, (1), 1, 1)
    P = model.num_prototypes
    proto_class_identity = np.argmax(model.prototype_class_identity.cpu().numpy(), axis=1)  # shape (P)
    proto_class_specific = np.full(P, class_specific)
    num_classes = model.num_classes
    if abstain_class:
        K = num_classes - 1
        assert K >= 2, "Abstention-push must have >= 2 classes not including abstain"
        # for the uncertainty prototypes, class_specific is False
        # for now assume that each class (inc. unc.) has P_per_class == P/num_classes
        P_per_class = P // num_classes
        proto_class_specific[K * P_per_class : P] = False
    else:
        K = num_classes

    # keep track of the input embedding closest to each prototype
    proto_dist_ = np.full(P, np.inf)  # saves the distances to prototypes (distance = 1-CosineSimilarities). shape (P)
    # save some information dynamically for each prototype
    # which are updated whenever a closer match to prototype is found
    occurrence_map_ = [None for _ in range(P)]  # saves the computed occurence maps. shape (P, 1, (T), H, W)
    # saves the input to prototypical layer (conv feature * occurrence map), shape (P, D)
    protoL_input_ = [None for _ in range(P)]
    # saves the input images with embeddings closest to each prototype. shape (P, 3, (To), Ho, Wo)
    image_ = [None for _ in range(P)]
    # saves the gt label. shape (P)
    gt_ = [None for _ in range(P)]
    # saves the prediction logits of cases seen. shape (P, K)
    pred_ = [None for _ in range(P)]
    # saves the filenames of cases closest to each prototype. shape (P)
    filename_ = [None for _ in range(P)]

    data_iter = iter(dataloader)
    iterator = tqdm(range(len(dataloader)), dynamic_ncols=True)
    for push_iter in iterator:
        data_sample = next(data_iter)

        x = data_sample["cine"]  # shape (B, 3, (To), Ho, Wo)
        if preprocess_input_function is not None:
            x = preprocess_input_function(x)

        # get the network outputs for this instance
        with torch.no_grad():
            x = x.cuda()
            (
                protoL_input_torch,
                proto_dist_torch,
                occurrence_map_torch,
                logits,
            ) = model.push_forward(x)
            pred_torch = logits.softmax(dim=1)

        # record down batch data as numpy arrays
        protoL_input = protoL_input_torch.detach().cpu().numpy()  # shape (B, P, D)
        proto_dist = proto_dist_torch.detach().cpu().numpy()  # shape (B, P)
        occurrence_map = occurrence_map_torch.detach().cpu().numpy()  # shape (B, P, 1, (T), H, W)
        # pred = pred_torch.detach().cpu().numpy() # shape (B, num_classes)
        pred = logits.detach().cpu().numpy()  # shape (B, num_classes)
        gt = data_sample["target_AS"].detach().cpu().numpy()  # shape (B)
        image = x.detach().cpu().numpy()  # shape (B, 3, (To), Ho, Wo)
        filename = data_sample["filename"]  # shape (B)

        # for each prototype, find the minimum distance and their indices
        for j in range(P):
            proto_dist_j = proto_dist[:, j]  # (B)
            if proto_class_specific[j]:
                # compare with only the images of the prototype's class
                proto_dist_j = np.ma.masked_array(proto_dist_j, gt != proto_class_identity[j])
                if proto_dist_j.mask.all():
                    # if none of the classes this batch are the class of interest, move on
                    continue
            proto_dist_j_min = np.amin(proto_dist_j)  # scalar

            # if the distance this batch is smaller than prev.best, save it
            if proto_dist_j_min <= proto_dist_[j]:
                a = np.argmin(proto_dist_j)
                proto_dist_[j] = proto_dist_j_min
                protoL_input_[j] = protoL_input[a, j]
                occurrence_map_[j] = occurrence_map[a, j]
                pred_[j] = pred[a]
                image_[j] = image[a]
                gt_[j] = gt[a]
                filename_[j] = filename[a]

    prototypes_similarity_to_src_ROIs = 1 - np.array(proto_dist_)  # invert distance to similarity  shape (P)
    prototypes_occurrence_maps = np.array(occurrence_map_)  # shape (P, 1, (T), H, W)
    prototypes_src_imgs = np.array(image_)  # shape (P, 3, (To), Ho, Wo)
    prototypes_gts = np.array(gt_)  # shape (P)
    prototypes_preds = np.array(pred_)  # shape (P, K)
    prototypes_filenames = np.array(filename_)  # shape (P)

    # save the prototype information in a pickle file
    prototype_data_dict = {
        "prototypes_filenames": prototypes_filenames,
        "prototypes_src_imgs": prototypes_src_imgs,
        "prototypes_gts": prototypes_gts,
        "prototypes_preds": prototypes_preds,
        "prototypes_occurrence_maps": prototypes_occurrence_maps,
        "prototypes_similarity_to_src_ROIs": prototypes_similarity_to_src_ROIs,
    }
    save_pickle(prototype_data_dict, f"{proto_epoch_dir}/prototypes_info.pickle")

    if replace_prototypes:
        protoL_input_ = np.array(protoL_input_)
        log("\tExecuting push ...")
        prototype_update = np.reshape(protoL_input_, tuple(prototype_shape))
        model.prototype_vectors.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).cuda())
    end = time.time()
    log("\tpush time: \t{0}".format(end - start))