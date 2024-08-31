import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

def _train_or_test(model, data_loader, optimizer, device, is_train=True):
    """Perform training or testing steps on given model and data loader."""
    device='cpu'
    model.to(device)
    if is_train:
        model.train()
    else:
        model.eval()
    
    total_loss = 0
    
    final_pred_targets = []
    final_pred_outputs = []
    
    # Process to handle the context managers for training and testing
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        # for X, _, _, y, bweight_pred, slice_weight in tqdm(data_loader, leave=False):
        for X, _, _, y, bweight_pred in tqdm(data_loader, leave=False):
            X, y = X.to(device), y.to(device)
            bweight_pred = bweight_pred.float().unsqueeze(1).to(device)
            # slice_weight = slice_weight.float().unsqueeze(1).to(device)
            y = y.float().unsqueeze(1)
            
            # Forward pass
            outputs = model(X)
            
            # bweight_pred = bweight_pred * slice_weight
                        
            # Compute loss
            loss = torch.nn.functional.binary_cross_entropy(outputs, y, weight = bweight_pred)
            total_loss += loss.item()
            
            # Collect data for statistics
            preds = outputs.round()
            final_pred_targets.extend(y.cpu().numpy())
            final_pred_outputs.extend(preds.detach().cpu().numpy())
            
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
    average_loss = total_loss / len(data_loader)
    
    balanced_accuracy = balanced_accuracy_score(final_pred_targets, final_pred_outputs)
    f1 = f1_score(final_pred_targets, final_pred_outputs)
    precision = precision_score(final_pred_targets, final_pred_outputs)
    recall = recall_score(final_pred_targets, final_pred_outputs)
    auc = roc_auc_score(final_pred_targets, final_pred_outputs)
    
    metrics = {'average_loss': average_loss, 
               'final_balanced_accuracy': balanced_accuracy,
               'final_f1': f1,
               'final_precision': precision,
               'final_recall': recall,
               'final_auc': auc,
            }
    
    return metrics


def train_step(model, data_loader, optimizer, device):
    """Train the model for one epoch."""
    train_metrics = _train_or_test(
        model, data_loader, optimizer, device, is_train=True
    )
    print(f"Train loss: {train_metrics['average_loss']:.5f}")
    print(f"Final Output - BAccuracy: {train_metrics['final_balanced_accuracy']*100:.2f}% | F1: {train_metrics['final_f1']*100:.2f}% | AUC: {train_metrics['final_auc']*100:.2f}%")
    return train_metrics

def test_step(model, data_loader, device):
    """Evaluate the model."""
    test_metrics = _train_or_test(
        model, data_loader, None, device, is_train=False
    )
    print(f"\nValidation loss: {test_metrics['average_loss']:.5f}")
    print(f"Final Output - BAccuracy: {test_metrics['final_balanced_accuracy']*100:.2f}% | F1: {test_metrics['final_f1']*100:.2f}% | AUC: {test_metrics['final_auc']*100:.2f}%")
    return test_metrics

# Function to evaluate the model on the test set
def evaluate_model(data_loader, model, device):
    model.eval()  # Set the model to evaluation mode
    final_pred_targets = []
    final_pred_outputs = []
    confusion_matrix = np.zeros((2, 2), dtype=int)
    with torch.no_grad():  # Turn off gradients for validation, saves memory and computations
        for X, _, _, y, _ in tqdm(data_loader, leave=False):
            images = X.to(device)
            y = y.float().unsqueeze(1).to(device)
            outputs = model(images)
            preds = outputs.round()
            final_pred_targets.extend(y.cpu().numpy())
            final_pred_outputs.extend(preds.detach().cpu().numpy())
            
            for i, l in enumerate(y.int()):
                confusion_matrix[l.item(), int(preds[i].item())] += 1


    balanced_accuracy = balanced_accuracy_score(final_pred_targets, final_pred_outputs)
    f1 = f1_score(final_pred_targets, final_pred_outputs)
    precision = precision_score(final_pred_targets, final_pred_outputs)
    recall = recall_score(final_pred_targets, final_pred_outputs)
    auc = roc_auc_score(final_pred_targets, final_pred_outputs)
    
    metrics = {'final_balanced_accuracy': balanced_accuracy,
               'final_f1': f1,
               'final_precision': precision,
               'final_recall': recall,
               'final_auc': auc,
            }
    
    return metrics, confusion_matrix

