import os
import pandas as pd
import matplotlib.pyplot as plt
import torch

# Function to save metrics to a CSV file
def save_metrics_to_csv(all_train_metrics, all_test_metrics, csv_path):
    """Save training and testing metrics to a CSV file."""
    df_train = pd.DataFrame(all_train_metrics)
    df_test = pd.DataFrame(all_test_metrics)
    df = pd.concat([df_train, df_test], axis=1, keys=['Train', 'Test'])
    df.to_csv(csv_path)

# Function to plot and save loss
def plot_and_save_loss(all_train_metrics, all_test_metrics, plot_path):
    train_losses = [metrics['average_loss'] for metrics in all_train_metrics]
    test_losses = [metrics['average_loss'] for metrics in all_test_metrics]

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(plot_path)
    plt.close()

# Function to save the model's state dictionary in size-controlled chunks
def save_model_in_chunks(model_state_dict, save_path, max_size_bytes=50 * 1024 * 1024):
    """Saves the model's state dictionary in chunks that do not exceed a specified size in bytes."""
    current_chunk = {}
    current_size = 0
    part = 1

    # Create the directory if it does not exist
    os.makedirs(save_path, exist_ok=True)

    for key, tensor in model_state_dict.items():
        tensor_size = tensor.element_size() * tensor.nelement()  # Calculate tensor size in bytes
        if current_size + tensor_size > max_size_bytes and current_chunk:
            # Save the current chunk if adding this tensor would exceed the max size
            chunk_filename = os.path.join(save_path, f'model_part_{part}.pth')
            torch.save(current_chunk, chunk_filename)
            # print(f'Saved {chunk_filename}')
            current_chunk = {}
            current_size = 0
            part += 1
        
        current_chunk[key] = tensor
        current_size += tensor_size

    # Save any remaining parameters
    if current_chunk:
        chunk_filename = os.path.join(save_path, f'model_part_{part}.pth')
        torch.save(current_chunk, chunk_filename)
        # print(f'Saved {chunk_filename}')

    print(f'\nModel state saved in {part} parts.')

# Function to load model state dictionary from chunks
def load_model_from_chunks(load_path):
    """Loads model state dictionary from chunks saved in separate files."""
    model_state_dict = {}
    chunk_files = sorted([f for f in os.listdir(load_path) if f.startswith('model_part_') and f.endswith('.pth')])

    for chunk_file in chunk_files:
        chunk_path = os.path.join(load_path, chunk_file)
        chunk_state_dict = torch.load(chunk_path)
        model_state_dict.update(chunk_state_dict)
    
    print(f'Model state reconstructed from {len(chunk_files)} parts.')
    return model_state_dict

# Function to set up directory paths for saving models, plots, and metrics
def setup_directories(base_path, experiment_model, experiment_run):
    """Create directory structure for saving models, plots, and metrics."""
    paths = {
        'plots': os.path.join(base_path, experiment_model, experiment_run, 'plots'),
        'weights': os.path.join(base_path, experiment_model, experiment_run, 'weights'),
        'metrics': os.path.join(base_path, experiment_model, experiment_run)
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths
