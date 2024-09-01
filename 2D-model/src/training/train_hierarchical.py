from tqdm import tqdm
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import numpy as np

def _adjust_weights(balanced_accuracies, exponent=5, target_sum=5):
    """
    Adjusts the weights based on the balanced accuracies.

    Args:
        balanced_accuracies (list): A list of balanced accuracies.
        exponent (int, optional): The exponent used for calculating the weights. Defaults to 5.
        target_sum (int, optional): The target sum of the scaled weights. Defaults to 2.

    Returns:
        list: A list of scaled weights.
    """
    # Calculate weights as the exponentiation of the inverse of the accuracies
    weights = [1.0 / (acc ** exponent + 1e-6) for acc in balanced_accuracies]
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    # Scale the normalized weights so that their sum equals the target_sum
    scaled_weights = [w * target_sum for w in normalized_weights]
    return scaled_weights

def _train_or_test(model, data_loader, optimizer, device, is_train=True, task_weights=None):
    model.to(device)
    if is_train:
        model.train()
    else:
        model.eval()
    
    num_tasks = model.num_tasks
    
    total_loss = 0
    
    task_losses = [0] * num_tasks  # Assuming 5 tasks

    final_pred_targets = [[] for _ in range(num_tasks)]
    final_pred_outputs = [[] for _ in range(num_tasks)]
    
    final_targets = []  # For calculating balanced accuracy for final output
    final_outputs = []  # For calculating balanced accuracy for final output
    
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for X, targets, bweights_chars, final_target, bweight, slice_weight in tqdm(data_loader, leave=False):  # Assuming final_target is for the final output
            X = X.to(device)
            # bweights_chars = [b.float().unsqueeze(1).to(device) for b in bweights_chars]
            bweights_chars = [b.float().to(device) for b in bweights_chars]
            # targets = [t.float().unsqueeze(1).to(device) for t in targets]
            targets = [t.long().to(device) for t in targets]
            # targets = [t - 1 for t in targets]  # Assuming targets are 1-indexed
            final_target = final_target.float().unsqueeze(1).to(device)
            bweight = bweight.float().unsqueeze(1).to(device)
            slice_weight = slice_weight.float().unsqueeze(1).to(device)
            
            bweight_pred = bweight * slice_weight
            
            final_output, task_outputs = model(X)
            
            loss = 0
            for i, (task_output, target, bweight_char) in enumerate(zip(task_outputs, targets, bweights_chars)):
                bweight_char = bweight_char[0][target] # Get the first element of the batch
                bweight_char = bweight_char * slice_weight
                
                # Compute loss for each task
                # task_loss = torch.nn.functional.cross_entropy(task_output, target, weight=bweight_char)
                task_loss = torch.nn.functional.cross_entropy(task_output, target, reduction='none')
                task_loss = (task_loss * bweight_char).mean()
                
                # Multiply the loss by the task weight
                if task_weights:
                    task_loss =  task_loss * task_weights[i]
                    
                task_losses[i] += task_loss.item()
                loss += task_loss

                # Compute accuracy for each task
                preds = task_output.argmax(dim=1)
                final_pred_targets[i].extend(target.cpu().numpy())
                final_pred_outputs[i].extend(preds.detach().cpu().numpy())   

            # Compute loss for final output
            final_loss = torch.nn.functional.binary_cross_entropy(final_output, final_target, weight=bweight)
            loss += final_loss
            
            # Compute accuracy for final output
            final_preds = final_output.round()
            final_targets.extend(final_target.cpu().numpy())
            final_outputs.extend(final_preds.detach().cpu().numpy())

            total_loss += loss.item()  # Sum up total loss
            
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    average_loss = total_loss / len(data_loader)
    task_losses = [task_loss / len(data_loader) for task_loss in task_losses]
    task_balanced_accuracies = [balanced_accuracy_score(targets, outputs) for targets, outputs in zip(final_pred_targets, final_pred_outputs)]
    final_balanced_accuracy = balanced_accuracy_score(final_targets, final_outputs)
    final_f1 = f1_score(final_targets, final_outputs)
    final_precision = precision_score(final_targets, final_outputs)
    final_recall = recall_score(final_targets, final_outputs)
    final_auc = roc_auc_score(final_targets, final_outputs)
    
    # return the metrics as a dictionary
    metrics = {'average_loss': average_loss, 
               'task_losses': task_losses,
               'task_balanced_accuracies': task_balanced_accuracies,
               'final_balanced_accuracy': final_balanced_accuracy,
               'final_f1': final_f1,
               'final_precision': final_precision,
               'final_recall': final_recall,
               'final_auc': final_auc,
            }
    
    # if is_train:
    #     return metrics
    # elif is_train is False:
    #     # Adjust task weights based on the validation balanced accuracies
    #     task_weights = _adjust_weights(task_balanced_accuracies, exponent=2, target_sum=5)
    #     return metrics, task_weights
    return metrics

def train_step(model, data_loader, optimizer, device, task_weights=None):
    """
    Train the model for one epoch.

    Args:
        model (torch.nn.Module): The model to be trained.
        data_loader (torch.utils.data.DataLoader): The data loader for training data.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        device (torch.device): The device to be used for training.
        task_weights (list): The weights for each task in the model.

    Returns:
        tuple: A tuple containing the training metrics and updated task weights.
    """
    train_metrics = _train_or_test(
        model, data_loader, optimizer, device, is_train=True, task_weights=task_weights
    )
    print(f"Train loss: {train_metrics['average_loss']:.5f}")
    for i, (loss, bal_acc) in enumerate(zip(train_metrics['task_losses'], train_metrics['task_balanced_accuracies']), 1):
        print(f"Task {i} - Loss: {loss:.2f}, Train Balanced Accuracy: {bal_acc*100:.2f}%")
    # Print the metrics for the final output
    print(f"Final Output - Train Balanced Accuracy: {train_metrics['final_balanced_accuracy']*100:.2f}%, Train F1: {train_metrics['final_f1']*100:.2f}%, Recall: {train_metrics['final_recall']*100:.2f}%, Precision: {train_metrics['final_precision']*100:.2f}%")
    return train_metrics


def test_step(model, data_loader, device, task_weights=None):
    """
    Evaluate the model on the test dataset.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        data_loader (torch.utils.data.DataLoader): The data loader for the test dataset.
        device (torch.device): The device to run the evaluation on.

    Returns:
        dict: A dictionary containing the evaluation metrics.
    """
    # Unpack the return values from _train_or_test function including the final output metrics
    test_metrics = _train_or_test(
        model, data_loader, None, device, is_train=False, task_weights=task_weights
    )
    print(f"Val loss: {test_metrics['average_loss']:.5f}")
    for i, (loss, bal_acc) in enumerate(zip(test_metrics['task_losses'], test_metrics['task_balanced_accuracies']), 1):
        print(f"Task {i} - Loss: {loss:.2f}, Train Balanced Accuracy: {bal_acc*100:.2f}%")
    # Print the metrics for the final output
    print(f"Final Output - Balanced Accuracy: {test_metrics['final_balanced_accuracy']*100:.2f}%, F1: {test_metrics['final_f1']*100:.2f}%, Recall: {test_metrics['final_recall']*100:.2f}%, Precision: {test_metrics['final_precision']*100:.2f}%")
    return test_metrics

# Function to evaluate the model on the test set
def evaluate_model(data_loader, model, device):
    model.eval()  # Set the model to evaluation mode
    
    final_pred_targets = [[] for _ in range(model.num_tasks)]
    final_pred_outputs = [[] for _ in range(model.num_tasks)]
    
    final_targets = []
    final_outputs = []
    
    confusion_matrix = np.zeros((2, 2), dtype=int)
    with torch.no_grad():  # Turn off gradients for validation, saves memory and computations
        for X, targets, _, final_target, _, _ in tqdm(data_loader, leave=False):  # Assuming final_target is for the final output
            X = X.to(device)
            targets = [t.long().to(device) for t in targets]
            # targets = [t - 1 for t in targets]  # Assuming targets are 1-indexed
            final_target = final_target.float().unsqueeze(1).to(device)
            
            final_output, task_outputs = model(X)
            
            for i, (task_output, target) in enumerate(zip(task_outputs, targets)):
                # Compute accuracy for each task
                preds = task_output.argmax(dim=1)
                final_pred_targets[i].extend(target.cpu().numpy())
                final_pred_outputs[i].extend(preds.detach().cpu().numpy())  
            
            preds = final_output.round()
            final_targets.extend(final_target.cpu().numpy())
            final_outputs.extend(preds.detach().cpu().numpy())
            
            for i, l in enumerate(final_target.int()):
                confusion_matrix[l.item(), int(preds[i].item())] += 1
    
    task_balanced_accuracies = [balanced_accuracy_score(targets, outputs) for targets, outputs in zip(final_pred_targets, final_pred_outputs)]
    final_balanced_accuracy = balanced_accuracy_score(final_targets, final_outputs)
    final_f1 = f1_score(final_targets, final_outputs)
    final_precision = precision_score(final_targets, final_outputs)
    final_recall = recall_score(final_targets, final_outputs)
    final_auc = roc_auc_score(final_targets, final_outputs)
    
    # return the metrics as a dictionary
    metrics = {'task_balanced_accuracies': task_balanced_accuracies,
               'final_balanced_accuracy': final_balanced_accuracy,
               'final_f1': final_f1,
               'final_precision': final_precision,
               'final_recall': final_recall,
               'final_auc': final_auc,
            }
    
    return metrics, confusion_matrix

def evaluate_model_by_nodule(model, data_loader, device, mode="median", decision_threshold=0.5, std_dev=1.2):
    model.to(device)
    model.eval()
    
    final_pred_targets = [[] for _ in range(model.num_tasks)]
    final_pred_outputs = [[] for _ in range(model.num_tasks)]
    
    final_targets = []
    final_outputs = []
    
    with torch.no_grad():
        for slices, task_labels, labels in tqdm(data_loader, leave=False):
            slices = slices.to(device)
            
            # Reshape slices if your model expects a single batch dimension
            if slices.dim() == 5:  # Assuming slices is (batch_size, num_slices, channels, height, width)
                slices = slices.view(-1, slices.size(2), slices.size(3), slices.size(4))  # Flatten the slices into one batch
            
            outputs, task_outputs = model(slices)
            
            for i, (task_output, task_label) in enumerate(zip(task_outputs, task_labels)):
                if task_output.ndim > 1 and task_output.shape[1] == 1:  # If model outputs a single probability per slice
                    task_output = task_output.squeeze(1)

                if mode == "median":
                    # Calculate the median prediction for the nodule
                    task_output = torch.median(task_output, dim=0)[0]

                preds = task_output.argmax()
                print(preds, task_label, preds.shape, task_label.shape)
                final_pred_targets[i].extend(task_label.numpy())
                final_pred_outputs[i].extend(preds.detach().cpu().numpy())  
            
            if outputs.ndim > 1 and outputs.shape[1] == 1:  # If model outputs a single probability per slice
                outputs = outputs.squeeze(1)

            if mode == "median":
                # Calculate the median prediction for the nodule
                outputs = outputs.median()

            # predictions = predictions.round()
            predictions = (outputs > decision_threshold).float()

            # Append the final prediction for the nodule
            final_targets.append(labels.numpy())
            final_outputs.append(predictions.cpu().numpy())
    print("test")
    task_balanced_accuracies = [balanced_accuracy_score(targets, outputs) for targets, outputs in zip(final_pred_targets, final_pred_outputs)]
    balanced_accuracy = balanced_accuracy_score(final_targets, final_outputs)
    f1 = f1_score(final_targets, final_outputs)
    precision = precision_score(final_targets, final_outputs)
    recall = recall_score(final_targets, final_outputs)
    auc = roc_auc_score(final_targets, final_outputs)
    
    # calculate confusion matrix
    confusion_matrix = np.zeros((2, 2), dtype=int)
    for i, l in enumerate(final_targets):
        confusion_matrix[int(l), int(final_outputs[i])] += 1
    
    metrics = {'final_balanced_accuracy': balanced_accuracy,
               'final_f1': f1,
               'final_precision': precision,
               'final_recall': recall,
               'final_auc': auc,
               'task_balanced_accuracies': task_balanced_accuracies
            }

    return metrics, confusion_matrix