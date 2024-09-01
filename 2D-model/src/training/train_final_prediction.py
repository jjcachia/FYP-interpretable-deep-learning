import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

def _train_or_test(model, data_loader, optimizer, device, is_train=True):
    """Perform training or testing steps on given model and data loader."""
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
            outputs = torch.sigmoid(outputs)
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
            outputs = torch.sigmoid(outputs)
            preds = outputs.round()
            final_pred_targets.extend(y.cpu().numpy())
            final_pred_outputs.extend(preds.detach().cpu().numpy())
            
            # for i, l in enumerate(y.int()):
            #     confusion_matrix[l.item(), int(preds[i].item())] += 1


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

def evaluate_model_by_nodule(model, data_loader, device, mode="median", decision_threshold=0.5, std_dev=1.2):
    model.to(device)
    model.eval()
    
    final_pred_targets = []
    final_pred_outputs = []
    
    with torch.no_grad():
        for slices, labels,_ in tqdm(data_loader, leave=False):
            slices = slices.to(device)
            
            # Reshape slices if your model expects a single batch dimension
            if slices.dim() == 5:  # Assuming slices is (batch_size, num_slices, channels, height, width)
                slices = slices.view(-1, slices.size(2), slices.size(3), slices.size(4))  # Flatten the slices into one batch
            
            predictions = model(slices)
            
            if predictions.ndim > 1 and predictions.shape[1] == 1:  # If model outputs a single probability per slice
                predictions = predictions.squeeze(1)

            if mode == "median":
                # Calculate the median prediction for the nodule
                predictions = predictions.median()
            elif mode == "mean":
                # Calculate the mean prediction for the nodule
                predictions = predictions.mean()
            elif mode == "gaussian":
                # Generate Gaussian weights centered at the central slice of the nodule
                num_slices = predictions.size(0)
                x = np.linspace(0, num_slices-1, num_slices)
                # mean = (num_slices - 1) / 2
                mean = num_slices / 2
                # std_dev = num_slices / std_dev
                std_dev = std_dev
                weights = norm.pdf(x, mean, std_dev)
                weights = torch.tensor(weights, dtype=torch.float32, device=device)
                weights = weights / weights.sum()
                predictions = (predictions * weights).sum()

            # predictions = predictions.round()
            predictions = (predictions > decision_threshold).float()

            # Append the final prediction for the nodule
            final_pred_targets.append(labels.numpy())
            final_pred_outputs.append(predictions.cpu().numpy())

    balanced_accuracy = balanced_accuracy_score(final_pred_targets, final_pred_outputs)
    f1 = f1_score(final_pred_targets, final_pred_outputs)
    precision = precision_score(final_pred_targets, final_pred_outputs)
    recall = recall_score(final_pred_targets, final_pred_outputs)
    auc = roc_auc_score(final_pred_targets, final_pred_outputs)
    
    # calculate confusion matrix
    confusion_matrix = np.zeros((2, 2), dtype=int)
    for i, l in enumerate(final_pred_targets):
        confusion_matrix[int(l), int(final_pred_outputs[i])] += 1
    
    metrics = {'final_balanced_accuracy': balanced_accuracy,
               'final_f1': f1,
               'final_precision': precision,
               'final_recall': recall,
               'final_auc': auc,
            }

    return metrics, confusion_matrix