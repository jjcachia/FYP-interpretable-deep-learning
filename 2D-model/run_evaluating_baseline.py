import os, time, gc, argparse, shutil
import pandas as pd
import torch, torch.utils.data, torchvision.transforms as transforms, torch.nn as nn

from src.utils.helpers import setup_directories, load_model_from_chunks, set_seed
from src.loaders._2D.dataloader import LIDCDataset
from src.models.base_model import construct_baseModel
from src.models.baseline_model import construct_baselineModel
from src.evaluation.evaluating import LIDCEvaluationDataset, evaluate_model_by_nodule

IMG_CHANNELS = 3
IMG_SIZE = 100
CHOSEN_CHARS = [False, True, False, True, True, False, False, True]

DEFAULT_BATCH_SIZE = 25
DEFAULT_EPOCHS = 100
DEFAULT_LEARNING_RATE = 0.00001

MODEL_DICT = {
    'baseline': construct_baselineModel,
    'base': construct_baseModel
}

def parse_args():
    parser = argparse.ArgumentParser(description="Train a deep learning model on the specified dataset.")
    parser.add_argument('--experiment_run', type=str, required=True, help='Identifier for the experiment run')
    
    parser.add_argument('--backbone', type=str, default='denseNet121', help='Feature Extractor Backbone to use')
    parser.add_argument('--model', type=str, default='base', help='Model to train')
    parser.add_argument('--weights', type=str, default='DEFAULT', help='Weights to use for the backbone model')
    
    parser.add_argument('--img_channels', type=int, default=IMG_CHANNELS, help='Number of channels in the input image')
    parser.add_argument('--img_size', type=int, default=IMG_SIZE, help='Size of the input image')
    
    parser.add_argument('--device', type=str, default='0', help='GPU device to use')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=DEFAULT_LEARNING_RATE, help='Learning rate for optimizer')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Get the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(script_dir, 'saved_models')
    
    # Define your experiment details
    experiment_model = args.model
    experiment_backbone = args.backbone
    experiment_run = args.experiment_run

    # Setup directories
    paths = setup_directories(base_path, experiment_model, experiment_backbone, experiment_run)
    best_model_path = os.path.join(paths['weights'], 'best_model.pth')
    test_metrics_path = os.path.join(paths['metrics'], 'test_metrics.csv')

    # Set the seed for reproducibility
    set_seed(27)
    
    # Check if CUDA is available
    print("#"*100 + "\n\n")
    if torch.cuda.is_available():
        print("CUDA is available. GPU devices:")
        # Loop through all available GPUs
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available. Only CPU is available.")
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:' + args.device) 
    print(f"Using device: {device}")

    ###############################################################################################################
    ###################################### Initialize the model ###################################################
    ###############################################################################################################

    if args.model not in MODEL_DICT:
        raise ValueError(f"Unsupported model name {args.model}")
    construct_Model = MODEL_DICT[args.model]
    
    # Create the model instance
    model = construct_Model(backbone_name=args.backbone, weights=args.weights)
    model.to(device)
    
    ###############################################################################################################
    ####################################### Evaluate the model ####################################################
    ###############################################################################################################
    print("\n\n" + "#"*100 + "\n\n")
    
    # test set
    labels_file = os.path.join(script_dir, 'dataset', '2D', 'Meta', 'slice_labels.csv')
    LIDC_testset = LIDCEvaluationDataset(labels_file=labels_file, indeterminate=False, transform=transforms.Compose([transforms.Grayscale(num_output_channels=IMG_CHANNELS), transforms.ToTensor()]))
    test_dataloader = torch.utils.data.DataLoader(LIDC_testset, batch_size=None, shuffle=False, num_workers=0) # Predict one nodule at a time
    
    # Evaluate the model on the test set
    model.load_state_dict(load_model_from_chunks(best_model_path))
    test_metrics, test_confusion_matrix = evaluate_model_by_nodule(model, test_dataloader, device, mode="median")
    print(f"Test Metrics with Median Aggregation:")
    print(test_metrics)
    print("Test Confusion Matrix:")
    print(test_confusion_matrix)

    test_metrics, test_confusion_matrix = evaluate_model_by_nodule(model, test_dataloader, device, mode="gaussian", std_dev=0.6)
    print(f"Test Metrics with Gaussian Aggregation and Standard Deviation of 0.6:")
    print(test_metrics)
    print("Test Confusion Matrix:")
    print(test_confusion_matrix)
    
    test_metrics, test_confusion_matrix = evaluate_model_by_nodule(model, test_dataloader, device, mode="gaussian", std_dev=0.8)
    print(f"Test Metrics with Gaussian Aggregation and Standard Deviation of 0.8:")
    print(test_metrics)
    print("Test Confusion Matrix:")
    print(test_confusion_matrix)
    
    test_metrics, test_confusion_matrix = evaluate_model_by_nodule(model, test_dataloader, device, mode="gaussian", std_dev=1.0)
    print(f"Test Metrics with Gaussian Aggregation and Standard Deviation of 1.0:")
    print(test_metrics)
    print("Test Confusion Matrix:")
    print(test_confusion_matrix)
    
    test_metrics, test_confusion_matrix = evaluate_model_by_nodule(model, test_dataloader, device, mode="gaussian", std_dev=1.2)
    print(f"Test Metrics with Gaussian Aggregation and Standard Deviation of 1.2:")
    print(test_metrics)
    print("Test Confusion Matrix:")
    print(test_confusion_matrix)
    
    test_metrics, test_confusion_matrix = evaluate_model_by_nodule(model, test_dataloader, device, mode="gaussian", std_dev=1.4)
    print(f"Test Metrics with Gaussian Aggregation and Standard Deviation of 1.4:")
    print(test_metrics)
    print("Test Confusion Matrix:")
    print(test_confusion_matrix)
    
    test_metrics, test_confusion_matrix = evaluate_model_by_nodule(model, test_dataloader, device, mode="gaussian", std_dev=1.4)
    print(f"Test Metrics with Gaussian Aggregation and Standard Deviation of 1.6:")
    print(test_metrics)
    print("Test Confusion Matrix:")
    print(test_confusion_matrix)
    
    test_metrics, test_confusion_matrix = evaluate_model_by_nodule(model, test_dataloader, device, mode="gaussian", std_dev=1.8)
    print(f"Test Metrics with Gaussian Aggregation and Standard Deviation of 1.8:")
    print(test_metrics)
    print("Test Confusion Matrix:")
    print(test_confusion_matrix)
    
    test_metrics, test_confusion_matrix = evaluate_model_by_nodule(model, test_dataloader, device, mode="gaussian", std_dev=2.0)
    print(f"Test Metrics with Gaussian Aggregation and Standard Deviation of 2.0:")
    print(test_metrics)
    print("Test Confusion Matrix:")
    print(test_confusion_matrix)
    
    # Save the test metrics to a CSV file
    df_test = pd.DataFrame([test_metrics])
    df_test.to_csv(test_metrics_path)

if __name__ == '__main__':
    main()
