import os, time, gc
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch, torch.utils.data, torchvision.transforms as transforms

from src.utils.helpers import save_metrics_to_csv, plot_and_save_loss, save_model_in_chunks, load_model_from_chunks, setup_directories
from src.dataloader import LIDCBinaryDataset
from src.train import train_step, test_step
from src.models.baseline_models import construct_baseModel


# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define your experiment details
experiment_model = 'base_model'
experiment_run = '1'
base_path = os.path.join(script_dir, 'saved_models')

# Setup directories
paths = setup_directories(base_path, experiment_model, experiment_run)
best_model_path = os.path.join(paths['weights'], 'best_model.pth')
metrics_path = os.path.join(paths['metrics'], 'metrics.csv')
plot_path = os.path.join(paths['plots'], 'loss_plot.png')

# Check if CUDA is available
print("#"*100 + "\n\n")
if torch.cuda.is_available():
	print("CUDA is available. GPU devices:")
	# Loop through all available GPUs
	for i in range(torch.cuda.device_count()):
		print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
	print("CUDA is not available. Only CPU is available.")
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

###############################################################################################################
#################################### Initialize the data loaders ##############################################
###############################################################################################################

print("\n\n" + "#"*100 + "\n\n")
from settings import chosen_chars, train_batch_size, test_batch_size, labels_file

img_size = 100
mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)

chosen_chars = [False, True, False, True, True, False, False, True, True]

preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=3), 
    transforms.Resize(256),  # First resize to larger dimensions
    transforms.CenterCrop(224),  # Then crop to 224x224
    transforms.ToTensor(),  # Convert to tensor (also scales pixel values to [0, 1])
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# train set
LIDC_trainset = LIDCBinaryDataset(labels_file=labels_file, chosen_chars= chosen_chars, auto_split=True, zero_indexed=False, 
                                                          transform=transforms.Compose([transforms.Grayscale(num_output_channels=3), 
                                                                    transforms.Resize(size=(img_size, img_size), interpolation=Image.BILINEAR), 
                                                                    transforms.ToTensor(), 
                                                                    # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                                                    transforms.Normalize(mean, std)
                                                                    ]),
                                                          train=True)
train_dataloader = torch.utils.data.DataLoader(LIDC_trainset, batch_size=train_batch_size, shuffle=True, num_workers=0)

# test set
LIDC_testset = LIDCBinaryDataset(labels_file=labels_file, chosen_chars= chosen_chars, auto_split=True, zero_indexed=False, 
                                                         transform=transforms.Compose([transforms.Grayscale(num_output_channels=3), 
                                                                    transforms.Resize(size=(img_size, img_size), interpolation=Image.BILINEAR), 
                                                                    transforms.ToTensor(), 
                                                                    # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                                                    transforms.Normalize(mean, std)
                                                                    ]), 
                                                         train=False)
test_dataloader = torch.utils.data.DataLoader(LIDC_testset, batch_size=test_batch_size, shuffle=True, num_workers=0)

batch_images = next(iter(train_dataloader))

print(f"Batch Size: {batch_images[0].shape[0]}, Number of Channels: {batch_images[0].shape[1]}, Image Size: {batch_images[0].shape[2]} x {batch_images[0].shape[3]} (NCHW)\n")
print(f"Number of Characteristics: {len(batch_images[1])}")


###############################################################################################################
####################################### Training the model ####################################################
###############################################################################################################
print("\n\n" + "#"*100 + "\n\n")
gc.collect()
torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()

# Set the number of epochs (we'll keep this small for faster training times)
epochs = 100

# Create the model instance
model = construct_baseModel()

# Print total number of parameters
total_params = sum(p.numel() for p in model.parameters())
print("Total number of parameters: ", total_params)

# Create a dummy input tensor of size [50, 3, 100, 100]
dummy_input = torch.randn(50, 3, 100, 100)

# Forward pass through the model with dummy input
features = model(dummy_input)

# Print output shapes to verify
print("Features shape:", features.shape)
