from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score

def _adjust_weights(balanced_accuracies, exponent=5, target_sum=2):
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
    """Perform training or testing steps on given model and data loader."""
    model.to(device)
    if is_train:
        model.train()
    else:
        model.eval()
    
    total_loss = 0
    
    task_losses = [0] * 5  # Assuming 5 tasks
    total_correct = [0] * 5  # For tasks
    total_samples = [0] * 5  # For tasks
    final_pred_targets = [[] for _ in range(5)]
    final_pred_outputs = [[] for _ in range(5)]
    
    final_correct = 0  # For final output
    final_samples = 0  # For final output
    final_targets = []  # For calculating balanced accuracy for final output
    final_outputs = []  # For calculating balanced accuracy for final output
    
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for X, targets, bweights_chars, final_target, bweight in tqdm(data_loader, leave=False):  # Assuming final_target is for the final output
            X = X.to(device)
            bweights_chars = [b.float().unsqueeze(1).to(device) for b in bweights_chars]
            targets = [t.float().unsqueeze(1).to(device) for t in targets]
            final_target = final_target.float().unsqueeze(1).to(device)
            bweight = bweight.float().unsqueeze(1).to(device)
            
            final_output, task_outputs = model(X)
            
            loss = 0
            for i, (task_output, target, bweight_char) in enumerate(zip(task_outputs, targets, bweights_chars)):
                # Compute loss for each task
                task_loss = torch.nn.functional.binary_cross_entropy(task_output, target, weight=bweight_char)
                if task_weights:
                    task_loss *= task_weights[i]
                task_losses[i] += task_loss.item()
                loss += task_loss

                # Compute accuracy for each task
                preds = task_output.round()
                total_correct[i] += (preds == target).sum().item()
                total_samples[i] += target.size(0)
                
                # Collect data for balanced accuracy for each task
                final_pred_targets[i].extend(target.cpu().numpy())
                final_pred_outputs[i].extend(preds.detach().cpu().numpy())

            # Compute loss for final output
            final_loss = torch.nn.functional.binary_cross_entropy(final_output, final_target, weight=bweight)
            loss += final_loss
            final_preds = final_output.round()
            final_correct += (final_preds == final_target).sum().item()
            final_samples += final_target.size(0)
            final_targets.extend(final_target.cpu().numpy())
            final_outputs.extend(final_preds.detach().cpu().numpy())

            total_loss += loss.item()  # Sum up total loss
            
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    average_loss = total_loss / len(data_loader)
    task_accuracies = [correct / samples for correct, samples in zip(total_correct, total_samples)]
    task_balanced_accuracies = [balanced_accuracy_score(targets, outputs) for targets, outputs in zip(final_pred_targets, final_pred_outputs)]
    final_accuracy = final_correct / final_samples
    final_balanced_accuracy = balanced_accuracy_score(final_targets, final_outputs)
    
    # return the metrics as a dictionary
    metrics = {'average_loss': average_loss, 
               'task_accuracies': task_accuracies, 
               'task_balanced_accuracies': task_balanced_accuracies, 
               'final_accuracy': final_accuracy, 
               'final_balanced_accuracy': final_balanced_accuracy }
    
    if is_train:
        # Adjust task weights based on the latest balanced accuracies
        task_weights = _adjust_weights(task_balanced_accuracies, exponent=5, target_sum=2)
        return metrics, task_weights
    elif is_train is False:
        return metrics 

def train_step(model, data_loader, optimizer, device, task_weights):
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
    train_metrics, task_weights = _train_or_test(
        model, data_loader, optimizer, device, is_train=True, task_weights=task_weights
    )
    print(f"Train loss: {train_metrics['average_loss']:.5f}")
    for i, (acc, bal_acc) in enumerate(zip(train_metrics['task_accuracies'], train_metrics['task_balanced_accuracies']), 1):
        print(f"Task {i} - Train Accuracy: {acc*100:.2f}%, Train Balanced Accuracy: {bal_acc*100:.2f}%")
    # Print the metrics for the final output
    print(f"Final Output - Train Accuracy: {train_metrics['final_accuracy']*100:.2f}%, Train Balanced Accuracy: {train_metrics['final_balanced_accuracy']*100:.2f}%")
    return train_metrics, task_weights


def test_step(model, data_loader, device):
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
        model, data_loader, None, device, is_train=False
    )
    print(f"\nTest loss: {test_metrics['average_loss']:.5f}")
    for i, (acc, bal_acc) in enumerate(zip(test_metrics['task_accuracies'], test_metrics['task_balanced_accuracies']), 1):
        print(f"Task {i} - Test Accuracy: {acc*100:.2f}%, Test Balanced Accuracy: {bal_acc*100:.2f}%")
    # Print the metrics for the final output
    print(f"Final Output - Test Accuracy: {test_metrics['final_accuracy']*100:.2f}%, Test Balanced Accuracy: {test_metrics['final_balanced_accuracy']*100:.2f}%")
    return test_metrics
