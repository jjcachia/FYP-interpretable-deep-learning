import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score

def _train_or_test(model, data_loader, optimizer, device, is_train=True):
    """Perform training or testing steps on given model and data loader."""
    model.to(device)
    if is_train:
        model.train()
    else:
        model.eval()
    
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    final_pred_targets = []
    final_pred_outputs = []
    
    # Process to handle the context managers for training and testing
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for X, _, _, y, bweight_pred in tqdm(data_loader, leave=False):
            X, y = X.to(device), y.to(device)
            bweight_pred = bweight_pred.float().unsqueeze(1).to(device)
            y = y.float().unsqueeze(1)
            
            # Forward pass
            outputs = model(X)
            
            # Compute loss
            loss = torch.nn.functional.binary_cross_entropy(outputs, y, weight = bweight_pred)
            total_loss += loss.item()
            
            # Compute accuracy
            preds = outputs.round()
            total_correct += (preds == y).sum().item()
            total_samples += y.size(0)

            # Collect data for balanced accuracy
            final_pred_targets.extend(y.cpu().numpy())
            final_pred_outputs.extend(preds.detach().cpu().numpy())
            
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
    average_loss = total_loss / len(data_loader)
    accuracy = total_correct / total_samples
    balanced_accuracy = balanced_accuracy_score(np.array(final_pred_targets).ravel(), np.array(final_pred_outputs).ravel())
    
    metrics = {'average_loss': average_loss, 
               'final_accuracy': accuracy, 
               'final_balanced_accuracy': balanced_accuracy}
    
    return metrics


def train_step(model, data_loader, optimizer, device, task_weights=None):
    """Train the model for one epoch."""
    train_metrics = _train_or_test(
        model, data_loader, optimizer, device, is_train=True
    )
    print(f"Train loss: {train_metrics['average_loss']:.5f}")
    print(f"Final Output - Train Accuracy: {train_metrics['final_accuracy']*100:.2f}%, Train Balanced Accuracy: {train_metrics['final_balanced_accuracy']*100:.2f}%")
    return train_metrics, task_weights

def test_step(model, data_loader, device):
    """Evaluate the model."""
    test_metrics = _train_or_test(
        model, data_loader, None, device, is_train=False
    )
    print(f"\nTest loss: {test_metrics['average_loss']:.5f}")
    print(f"Final Output - Test Accuracy: {test_metrics['final_accuracy']*100:.2f}%, Test Balanced Accuracy: {test_metrics['final_balanced_accuracy']*100:.2f}%")
    return test_metrics
