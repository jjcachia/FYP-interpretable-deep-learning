import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
from tqdm import tqdm

from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from scipy.stats import norm

class LIDCEvaluationDataset(Dataset):
    def __init__(self, labels_file, transform=None, chosen_chars=None, indeterminate=True, validation_split=0.10, test_split=0.10):
        all_labels = pd.read_csv(labels_file)
        self.transform = transform
        
        # Preprocess the labels
        all_labels['Subtlety'] = all_labels['Subtlety'].replace({1: 0, 2: 0, 3: 0, 4: 1, 5: 1})
        all_labels.drop(columns=['Internalstructure'], inplace=True)
        all_labels['Calcification'] = all_labels['Calcification'].replace({1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 1})
        all_labels['Sphericity'] = all_labels['Sphericity'].replace({1: 0, 2: 0, 3: 0, 4: 1, 5: 1})
        all_labels['Margin'] = all_labels['Margin'].replace({1: 0, 2: 0, 3: 0, 4: 1, 5: 1})
        all_labels['Lobulation'] = all_labels['Lobulation'].replace({1: 0, 2: 1, 3: 1, 4: 1, 5: 1})
        all_labels['Spiculation'] = all_labels['Spiculation'].replace({1: 0, 2: 1, 3: 1, 4: 1, 5: 1})
        all_labels['Texture'] = all_labels['Texture'].replace({1: 0, 2: 0, 3: 0, 4: 0, 5: 1})
        all_labels['Diameter'] = all_labels['Diameter'].replace({1: 0, 2: 0, 3: 1, 4: 1, 5: 1})
        if indeterminate:
            all_labels['Malignancy'] = all_labels['Malignancy'].replace({1: 0, 2: 0, 3: 1, 4: 2, 5: 2})
        else:
            all_labels['Malignancy'] = all_labels['Malignancy'].replace({1: 0, 2: 0, 3: 0, 4: 1, 5: 1})
        
        # Extract patient identifiers from the image directory paths
        all_labels['patient_id'] = all_labels['image_dir'].apply(lambda x: os.path.basename(os.path.dirname(os.path.dirname(x))))
        
        # Get the test labels
        random_state = 27
        unique_patients = all_labels['patient_id'].unique()    
        _, temp_patients = train_test_split(unique_patients, test_size=(validation_split + test_split), random_state=random_state)
        _, test_patients = train_test_split(temp_patients, test_size=(test_split / (validation_split + test_split)), random_state=random_state)
        test_labels = all_labels[all_labels['patient_id'].isin(test_patients)]
        
        # Group the test labels by nodule
        # test_labels['nodule_id'] = test_labels['image_dir'].apply(lambda x: os.path.basename(os.path.dirname(os.path.dirname(x))+'-'+os.path.basename(os.path.dirname(x))))
        test_labels.loc[:, 'nodule_id'] = test_labels['image_dir'].apply(lambda x: os.path.basename(os.path.dirname(os.path.dirname(x))+'-'+os.path.basename(os.path.dirname(x))))

        self.nodule_labels = test_labels.groupby('nodule_id')
        self.nodule_keys = list(self.nodule_labels.groups.keys())

    def __len__(self):
        return len(self.nodule_labels)

    def __getitem__(self, idx):
        nodule_key = self.nodule_keys[idx]
        nodule_data = self.nodule_labels.get_group(nodule_key)
        
        # Load all slices for the nodule and the label of the nodule
        images = [np.load(row['image_dir']) for _, row in nodule_data.iterrows()]
        # TODO: use chosen_chars to select only the chosen characteristics
        # labels = nodule_data.iloc[0][['Diameter', 'Subtlety', 'Calcification', 'Sphericity', 'Margin', 'Lobulation', 'Spiculation', 'Texture', 'Malignancy']].values
        
        # Convert images if a transform is specified
        if self.transform:
            images = [self.transform(Image.fromarray(img)) for img in images]
        
        final_pred_label = nodule_data.iloc[0]['Malignancy']  # Assuming 'Malignancy' is the last label

        return torch.stack(images), final_pred_label

def evaluate_model_by_nodule(model, data_loader, device):
    model.to(device)
    model.eval()
    
    final_pred_targets = []
    final_pred_outputs = []
    
    with torch.no_grad():
        for slices, labels in tqdm(data_loader, leave=False):
            slices = slices.to(device)
            
            # Reshape slices if your model expects a single batch dimension
            if slices.dim() == 5:  # Assuming slices is (batch_size, num_slices, channels, height, width)
                slices = slices.view(-1, slices.size(2), slices.size(3), slices.size(4))  # Flatten the slices into one batch
            
            predictions = model(slices)
            
            if predictions.ndim > 1 and predictions.shape[1] == 1:  # If model outputs a single probability per slice
                predictions = predictions.squeeze(1)

            # Calculate the median prediction for the nodule
            median_prediction = predictions.median()
            
            median_prediction = median_prediction.round()
            
            # Append the final prediction for the nodule
            final_pred_targets.append(labels.numpy())
            final_pred_outputs.append(median_prediction.cpu().numpy())

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

def _evaluate_model_by_nodule(model, data_loader, device):
    model.to(device)
    model.eval()
    
    final_pred_targets = []
    final_pred_outputs = []
    
    with torch.no_grad():
        for slices, labels in tqdm(data_loader, leave=False):
            slices = slices.to(device)
            
            # Reshape slices if your model expects a single batch dimension
            if slices.dim() == 5:  # Assuming slices is (batch_size, num_slices, channels, height, width)
                slices = slices.view(-1, slices.size(2), slices.size(3), slices.size(4))  # Flatten the slices into one batch
            
            predictions = model(slices)
            
            if predictions.ndim > 1 and predictions.shape[1] == 1:  # If model outputs a single probability per slice
                predictions = predictions.squeeze(1)

            # Calculate the median prediction for the nodule
            median_prediction = predictions.median()
            
            median_prediction = median_prediction.round()
            
            # Append the final prediction for the nodule
            final_pred_targets.append(labels.numpy())
            final_pred_outputs.append(median_prediction.cpu().numpy())

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


def evaluate_model_by_nodule(model, data_loader, device, mode="median", decision_threshold=0.5):
    model.to(device)
    model.eval()
    
    final_pred_targets = []
    final_pred_outputs = []
    
    with torch.no_grad():
        for slices, labels in tqdm(data_loader, leave=False):
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
                mean = num_slices / 2
                std_dev = num_slices / 5
                weights = norm.pdf(x, mean, std_dev)
                weights = torch.tensor(weights, dtype=torch.float32, device=device)
                weights = weights / weights.sum()
                predictions = (predictions * weights).sum()
            # elif mode == "max_quarter":
            
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