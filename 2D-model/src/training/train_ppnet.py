from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim import optimizer
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, roc_auc_score

def _adjust_weights(task_losses, exponent=2, target_sum=5):
    """
    Adjusts the weights based on the task losses, using the sum of all task losses for normalization.
    
    Args:
        task_losses (list): List of losses for each task.
        exponent (int): The exponent used for calculating the inverse weights. Defaults to 2.
        target_sum (int): The total sum to which the weights should scale. Defaults to 5.

    Returns:
        list: A list of adjusted weights for each task.
    """
    # Calculate the total sum of all task losses
    total_loss = sum(task_losses)
    # Normalize each loss by the total sum of losses
    normalized_losses = [loss / total_loss for loss in task_losses] if total_loss > 0 else [0] * len(task_losses)
    # Calculate weights using the normalized losses
    weights = [1.0 / ((1.0 - loss) ** exponent + 1e-6) for loss in normalized_losses]
    total_weight = sum(weights)
    scaled_weights = [w / total_weight * target_sum for w in weights]
    return scaled_weights

def _train_or_test(model, data_loader, optimizer, device, is_train=True, use_l1_mask=True, coefs=None, task_weights=None):
    model.to(device)
    if is_train:
        model.train()
    else:
        model.eval()
        
    num_characteristics = model.num_characteristics
    
    total_loss = 0.0
    
    task_total_losses = [0.0] * num_characteristics
    task_cross_entropy = [0.0] * num_characteristics
    task_cluster_cost = [0.0] * num_characteristics
    task_separation_cost = [0.0] * num_characteristics
    task_l1 = [0.0] * num_characteristics
    total_correct = [0] * num_characteristics  # For tasks
    total_samples = [0] * num_characteristics  # For tasks
    final_pred_targets = [[] for _ in range(num_characteristics)] # For calculating balanced accuracy for each characteristic
    final_pred_outputs = [[] for _ in range(num_characteristics)] # For calculating balanced accuracy for each characteristic
    
    final_total_loss = 0.0
    final_correct = 0
    final_samples = 0
    final_targets = []  # For calculating balanced accuracy for final output
    final_outputs = []  # For calculating balanced accuracy for final output
    
    n_batches = 0
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for X, targets, bweights_chars, final_target, bweight in tqdm(data_loader, leave=False):
            X = X.to(device)
            bweights_chars = [b.float().to(device) for b in bweights_chars]            
            targets = [t.squeeze().to(device) for t in targets]
            final_target = final_target.float().unsqueeze(1).to(device)
            bweight = bweight.float().unsqueeze(1).to(device)
            
            final_output, task_outputs, min_distances = model(X)
            
            batch_loss = 0.0
            for i, (task_output, min_distance, target, bweight_char) in enumerate(zip(task_outputs, min_distances, targets, bweights_chars)):
                # Get the prototype identity for each characteristic
                prototype_char_identity = model.prototype_class_identity[i].to(device)
                
                # Get the max distance between prototypes
                max_dist = (model.prototype_shape[1] * model.prototype_shape[2] * model.prototype_shape[3])
                
                # Compute cross entropy cost for each characteristic
                cross_entropy = torch.nn.functional.cross_entropy(task_output, target, weight=bweight_char[0])
                cross_entropy = cross_entropy * (coefs['crs_ent'] if coefs else 1)
                
                # Compute cluster cost for each characteristic
                prototypes_of_correct_class = torch.t(prototype_char_identity[:,target]).to(device)    # batch_size * num_prototypes
                inverted_distances, _ = torch.max((max_dist - min_distance) * prototypes_of_correct_class, dim=1)
                cluster_cost = torch.mean((max_dist - inverted_distances) * bweight_char[range(bweight_char.size(0)), target]) # Increase the distance between the prototypes of the same class
                cluster_cost = cluster_cost * (coefs['clst'] if coefs else 1)
                
                # Compute separation cost for each characteristic
                prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                inverted_distances_to_nontarget_prototypes, _ = torch.max((max_dist - min_distance) * prototypes_of_wrong_class, dim=1)
                separation_cost = torch.mean((max_dist - inverted_distances_to_nontarget_prototypes) * bweight_char[range(bweight_char.size(0)), 1 - target]) # Decrease the distance between the prototypes of different classes
                separation_cost = separation_cost * (coefs['sep'] if coefs else 1)
                
                # Compute l1 regularization for each characteristic
                if use_l1_mask:
                    l1_mask = 1 - torch.t(prototype_char_identity).to(device)
                    l1 = (model.task_specific_classifier[i].weight * l1_mask).norm(p=1)
                else:
                    l1 = model.task_specific_classifier[i].weight.norm(p=1) 
                l1 = l1 * (coefs['l1'] if coefs else 1)
                
                # Update the different task losses for each characteristic
                task_cross_entropy[i] += cross_entropy.item()
                task_cluster_cost[i] += cluster_cost.item()
                task_separation_cost[i] += separation_cost.item()
                task_l1[i] += l1.item()
                
                # Compute the total loss for each characteristic
                task_loss = cross_entropy + cluster_cost + separation_cost + l1
                
                # Update the task total losses
                task_total_losses[i] += task_loss.item()
                
                # Apply task weights if provided
                if task_weights:
                    task_loss *= task_weights[i]
                
                # Update the total loss for the batch
                batch_loss += task_loss
                
                # Compute accuracy for each characteristic
                preds = task_output.argmax(dim=1)
                total_correct[i] += (preds == target).sum().item()
                total_samples[i] += target.size(0)
                
                # Collect data for balanced accuracy for each characteristic
                final_pred_targets[i].extend(target.cpu().numpy())
                final_pred_outputs[i].extend(preds.detach().cpu().numpy())                

            # Compute binary cross entropy loss for final output
            final_loss = torch.nn.functional.binary_cross_entropy(final_output, final_target, weight=bweight)
            batch_loss += final_loss
            final_total_loss += final_loss.item()
            
            # Compute statistics for final accuracy
            final_preds = final_output.round()
            final_correct += (final_preds == final_target).sum().item()
            final_samples += final_target.size(0)
            final_targets.extend(final_target.cpu().numpy())
            final_outputs.extend(final_preds.detach().cpu().numpy())
            
            total_loss += batch_loss.item()  # Sum up total loss
            
            # compute gradient and do SGD step
            if is_train:
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                
            n_batches += 1
    
    average_loss = total_loss / n_batches
    
    task_losses = [t / n_batches for t in task_total_losses]
    task_cross_entropy = [t / n_batches for t in task_cross_entropy]
    task_cluster_cost = [t / n_batches for t in task_cluster_cost]
    task_separation_cost = [t / n_batches for t in task_separation_cost]
    task_l1 = [t / n_batches for t in task_l1]
    task_accuracies = [correct / samples for correct, samples in zip(total_correct, total_samples)]
    task_balanced_accuracies = [balanced_accuracy_score(targets, outputs) for targets, outputs in zip(final_pred_targets, final_pred_outputs)]
    
    final_loss = final_total_loss / n_batches
    final_accuracy = final_correct / final_samples
    final_balanced_accuracy = balanced_accuracy_score(final_targets, final_outputs)
    final_f1 = f1_score(final_targets, final_outputs)
    # final_precision = precision_score(final_targets, final_outputs)
    final_recall = recall_score(final_targets, final_outputs)
    final_auc = roc_auc_score(final_targets, final_outputs)

    # return the metrics as a dictionary
    metrics = {'average_loss': average_loss, 
               'task_losses': task_losses,
               'task_accuracies': task_accuracies, 
               'task_balanced_accuracies': task_balanced_accuracies, 
               'task_cross_entropy': task_cross_entropy,
               'task_cluster_cost': task_cluster_cost,
               'task_separation_cost': task_separation_cost,
               'task_l1': task_l1,
               'final_loss': final_loss,
               'final_accuracy': final_accuracy,
               'final_balanced_accuracy': final_balanced_accuracy,
               'final_f1': final_f1,
               # 'final_precision': final_precision,
               'final_recall': final_recall,
               'final_auc': final_auc
            }
    
    if is_train:
        task_weights = _adjust_weights(task_losses, exponent=5, target_sum=4)
        return metrics, task_weights
    else:
        return metrics

def train_ppnet(model, data_loader, optimizer, device, use_l1_mask=True, coefs=None, task_weights=None):
    train_metrics, task_weights = _train_or_test(model, data_loader, optimizer, device, is_train=True, use_l1_mask=use_l1_mask, coefs=coefs, task_weights=task_weights)
    print("\nFinal Train Metrics:")
    print(f"Total Loss: {train_metrics['average_loss']:.5f}")
    for i, (bal_acc, task_loss, task_ce, task_cc, task_sc) in enumerate(zip(train_metrics['task_balanced_accuracies'], train_metrics['task_losses'], train_metrics['task_cross_entropy'], train_metrics['task_cluster_cost'], train_metrics['task_separation_cost']), 1):
        print(f"Characteristic {i}     - Task Loss: {task_loss:.2f}, Cross Entropy: {task_ce:.2f}, Cluster Cost: {task_cc:.2f}, Separation Cost: {task_sc:.2f}, Balanced Accuracy: {bal_acc*100:.2f}%")
    # Print the metrics for the final output
    print(f"Malignancy Prediction - Binary Cross Entropy Loss: {train_metrics['final_loss']:.2f}, Balanced Accuracy: {train_metrics['final_balanced_accuracy']*100:.2f}%, F1 Score: {train_metrics['final_f1']*100:.2f}%")
    return train_metrics, task_weights

def test_ppnet(model, data_loader, device, use_l1_mask=True, coefs=None, task_weights=None):
    test_metrics = _train_or_test(model, data_loader, None, device, is_train=False, use_l1_mask=use_l1_mask, coefs=coefs, task_weights=task_weights)
    print("\nFinal Test Metrics:")
    print(f"Total Loss: {test_metrics['average_loss']:.5f}")
    for i, (bal_acc, task_loss, task_ce, task_cc, task_sc) in enumerate(zip(test_metrics['task_balanced_accuracies'], test_metrics['task_losses'], test_metrics['task_cross_entropy'], test_metrics['task_cluster_cost'], test_metrics['task_separation_cost']), 1):
        print(f"Characteristic {i}     - Task Loss: {task_loss:.2f}, Cross Entropy: {task_ce:.2f}, Cluster Cost: {task_cc:.2f}, Separation Cost: {task_sc:.2f}, Balanced Accuracy: {bal_acc*100:.2f}%")
    # Print the metrics for the final output
    print(f"Malignancy Prediction - Binary Cross Entropy Loss: {test_metrics['final_loss']:.2f}, Balanced Accuracy: {test_metrics['final_balanced_accuracy']*100:.2f}%, F1 Score: {test_metrics['final_f1']*100:.2f}%")
    return test_metrics
            
def last_only(model):
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.add_on_layers.parameters():
        p.requires_grad = False
    model.prototype_vectors.requires_grad = False
    for p in model.task_specific_classifier.parameters():
        p.requires_grad = True
    for p in model.final_classifier.parameters():
        p.requires_grad = True # was true

def warm_only(model):
    if model.features.encoder is not None:
        for p in model.features.encoder.parameters():
            p.requires_grad = False
        for p in model.features.adaptation_layers.parameters():
            p.requires_grad = True
        for p in model.features.fpn.parameters():
            p.requires_grad = True
    else:
        for p in model.features.parameters():
            p.requires_grad = False
    for p in model.add_on_layers.parameters():
        p.requires_grad = True
    model.prototype_vectors.requires_grad = True
    for p in model.task_specific_classifier.parameters():
        p.requires_grad = False
    for p in model.final_classifier.parameters():
        p.requires_grad = False
        
def joint(model):
    for p in model.features.parameters():
        p.requires_grad = True
    for p in model.add_on_layers.parameters():
        p.requires_grad = True
    model.prototype_vectors.requires_grad = True
    for p in model.task_specific_classifier.parameters():
        p.requires_grad = True
    for p in model.final_classifier.parameters():
        p.requires_grad = True
    