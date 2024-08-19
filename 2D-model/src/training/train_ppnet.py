from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim import optimizer
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

def _train_or_test(model, data_loader, optimizer, device, is_train=True, use_l1_mask=True, coefs=None, task_weights=None):
    model.to(device)
    if is_train:
        model.train()
    else:
        model.eval()
    
    total_loss = 0
    
    total_correct = [0] * 5  # For tasks
    total_samples = [0] * 5  # For tasks
    
    task_cross_entropy = [0.0] * 5
    task_cluster_cost = [0.0] * 5
    task_separation_cost = [0.0] * 5
    task_avg_separation_cost = [0.0] * 5
    task_l1 = [0.0] * 5
    final_pred_targets = [[] for _ in range(5)]
    final_pred_outputs = [[] for _ in range(5)]
    
    final_correct = 0  # For final output
    final_samples = 0  # For final output
    final_targets = []  # For calculating balanced accuracy for final output
    final_outputs = []  # For calculating balanced accuracy for final output
    
    n_batches = 0
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for X, targets, bweights_chars, final_target, bweight in tqdm(data_loader, leave=False):
            X = X.to(device)
            bweights_chars = [b.float().to(device) for b in bweights_chars]
            
            targets2 = [F.one_hot(t.squeeze(), num_classes=2).float().to(device) for t in targets]
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

                # Compute cluster cost for each characteristic
                prototypes_of_correct_class = torch.t(prototype_char_identity[:,target]).to(device)    # batch_size * num_prototypes
                inverted_distances, _ = torch.max((max_dist - min_distance) * prototypes_of_correct_class, dim=1)
                cluster_cost = torch.mean(max_dist - inverted_distances) # Increase the distance between the prototypes of the same class

                # Compute separation cost for each characteristic
                prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                inverted_distances_to_nontarget_prototypes, _ = torch.max((max_dist - min_distance) * prototypes_of_wrong_class, dim=1)
                separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes) # Decrease the distance between the prototypes of different classes

                # Compute average separation cost for each characteristic
                avg_separation_cost = torch.sum(min_distance * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class, dim=1)
                avg_separation_cost = torch.mean(avg_separation_cost)
                
                # Compute l1 regularization for each characteristic
                if use_l1_mask:
                    l1_mask = 1 - torch.t(prototype_char_identity).to(device)
                    l1 = (model.task_specific_classifier[i].weight * l1_mask).norm(p=1)
                else:
                    l1 = model.task_specific_classifier[i].weight.norm(p=1) 
                    
                # Compute accuracy for each characteristic
                preds = task_output.argmax(dim=1)
                total_correct[i] += (preds == target).sum().item()
                total_samples[i] += target.size(0)
                
                # Collect data for balanced accuracy for each characteristic
                final_pred_targets[i].extend(target.cpu().numpy())
                final_pred_outputs[i].extend(preds.detach().cpu().numpy())

                task_cross_entropy[i] += cross_entropy.item()
                task_cluster_cost[i] += cluster_cost.item()
                task_separation_cost[i] += separation_cost.item()
                task_avg_separation_cost[i] += avg_separation_cost.item()
                task_l1[i] += l1
                
                batch_loss += (coefs['crs_ent'] * cross_entropy + 
                               coefs['clst'] * cluster_cost + 
                               coefs['sep'] * separation_cost +
                               coefs['l1'] * l1)

            # Compute binary cross entropy loss for final output
            final_loss = torch.nn.functional.binary_cross_entropy(final_output, final_target, weight=bweight)
            batch_loss += final_loss
            
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
    
    # TODO: Add the seperate characteristic losses to the return dictionary, and include final F1 score, final precision, final recall, and final AUC
    average_loss = total_loss / n_batches
    task_accuracies = [correct / samples for correct, samples in zip(total_correct, total_samples)]
    task_balanced_accuracies = [balanced_accuracy_score(targets, outputs) for targets, outputs in zip(final_pred_targets, final_pred_outputs)]
    final_accuracy = final_correct / final_samples
    final_balanced_accuracy = balanced_accuracy_score(final_targets, final_outputs)
    # final_f1 = f1_score(final_targets, final_outputs)
    # final_precision = precision_score(final_targets, final_outputs)
    # final_recall = recall_score(final_targets, final_outputs)
    # final_auc = roc_auc_score(final_targets, final_outputs)
    
    # task_cross_entropy = [t / n_batches for t in task_cross_entropy]
    # task_cluster_cost = [t / n_batches for t in task_cluster_cost]
    # task_separation_cost = [t / n_batches for t in task_separation_cost]
    # task_avg_separation_cost = [t / n_batches for t in task_avg_separation_cost]
    # task_l1 = [t / n_batches for t in task_l1]
    
    # return the metrics as a dictionary
    metrics = {'average_loss': average_loss, 
               'task_accuracies': task_accuracies, 
               'task_balanced_accuracies': task_balanced_accuracies, 
               'final_accuracy': final_accuracy, 
               'final_balanced_accuracy': final_balanced_accuracy}
               # 'task_cross_entropy': task_cross_entropy,
               # 'task_cluster_cost': task_cluster_cost,
               # 'task_separation_cost': task_separation_cost,
               # 'task_avg_separation_cost': task_avg_separation_cost,
               # 'task_l1': task_l1}
    
    if is_train:
        return metrics
    else:
        return metrics

def train_ppnet(model, data_loader, optimizer, device, use_l1_mask=True, coefs=None, task_weights=None):
    train_metrics = _train_or_test(model, data_loader, optimizer, device, is_train=True, use_l1_mask=use_l1_mask, coefs=coefs, task_weights=task_weights)
    print(f"Train loss: {train_metrics['average_loss']:.5f}")
    for i, (acc, bal_acc) in enumerate(zip(train_metrics['task_accuracies'], train_metrics['task_balanced_accuracies']), 1):
        print(f"Task {i} - Train Accuracy: {acc*100:.2f}%, Train Balanced Accuracy: {bal_acc*100:.2f}%")
    # Print the metrics for the final output
    print(f"Final Output - Train Accuracy: {train_metrics['final_accuracy']*100:.2f}%, Train Balanced Accuracy: {train_metrics['final_balanced_accuracy']*100:.2f}%")
    return train_metrics

def test_ppnet(model, data_loader, device, use_l1_mask=True, coefs=None, task_weights=None):
    test_metrics = _train_or_test(model, data_loader, None, device, is_train=False, use_l1_mask=use_l1_mask, coefs=coefs, task_weights=task_weights)
    print(f"Test loss: {test_metrics['average_loss']:.5f}")
    for i, (acc, bal_acc) in enumerate(zip(test_metrics['task_accuracies'], test_metrics['task_balanced_accuracies']), 1):
        print(f"Task {i} - Test Accuracy: {acc*100:.2f}%, Test Balanced Accuracy: {bal_acc*100:.2f}%")
    # Print the metrics for the final output
    print(f"Final Output - Test Accuracy: {test_metrics['final_accuracy']*100:.2f}%, Test Balanced Accuracy: {test_metrics['final_balanced_accuracy']*100:.2f}%")
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
        p.requires_grad = False # was true

def warm_only(model):
    for p in model.features.encoder.parameters():
        p.requires_grad = False
    for p in model.features.adaptation_layers.parameters():
        p.requires_grad = True
    for p in model.features.fpn.parameters():
        p.requires_grad = True
    # for p in model.features.parameters():
    #     p.requires_grad = False
    for p in model.add_on_layers.parameters():
        p.requires_grad = True
    model.prototype_vectors.requires_grad = True
    for p in model.task_specific_classifier.parameters():
        p.requires_grad = True
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
    