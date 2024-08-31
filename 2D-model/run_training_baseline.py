import os, time, gc, argparse, shutil
import pandas as pd
import torch, torch.utils.data, torchvision.transforms as transforms, torch.nn as nn

from src.utils.helpers import save_metrics_to_csv, plot_and_save_loss, save_model_in_chunks, setup_directories, load_model_from_chunks, set_seed
from src.loaders._3D.dataloader import LIDCDataset
from src.training.train_final_prediction import train_step, test_step, evaluate_model
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
    
    # Define your experiment details
    base_path = os.path.join(script_dir, 'saved_models')
    experiment_model = args.model
    experiment_backbone = args.backbone
    experiment_run = args.experiment_run

    # Setup directories
    paths = setup_directories(base_path, experiment_model, experiment_backbone, experiment_run)
    best_model_path = os.path.join(paths['weights'], 'best_model.pth')
    metrics_path = os.path.join(paths['metrics'], 'metrics.csv')
    test_metrics_path = os.path.join(paths['metrics'], 'test_metrics.csv')
    plot_path = os.path.join(paths['plots'], 'loss_plot.png')

    # Set the seed for reproducibility
    set_seed(27)
    
    # Save the script to the experiment directory
    shutil.copy(__file__, os.path.join(paths['scripts'], 'main.py'))
    
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
    #################################### Initialize the data loaders ##############################################
    ###############################################################################################################
    print("\n\n" + "#"*100 + "\n\n")

    # labels_file = './dataset/Meta/meta_info_old.csv'
    labels_file = os.path.join(script_dir, 'dataset', '3D', 'Meta', 'volume_labels.csv')
    
    # train set
    LIDC_trainset = LIDCDataset(labels_file=labels_file, chosen_chars=CHOSEN_CHARS, indeterminate=False, transform=transforms.Compose([transforms.Grayscale(num_output_channels=IMG_CHANNELS), transforms.ToTensor()]), split='train')
    train_dataloader = torch.utils.data.DataLoader(LIDC_trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # validation set
    LIDC_valset = LIDCDataset(labels_file=labels_file, chosen_chars=CHOSEN_CHARS, indeterminate=False, transform=transforms.Compose([transforms.Grayscale(num_output_channels=IMG_CHANNELS), transforms.ToTensor()]), split='val')
    val_dataloader = torch.utils.data.DataLoader(LIDC_valset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    # test set
    LIDC_testset = LIDCDataset(labels_file=labels_file, chosen_chars=CHOSEN_CHARS, indeterminate=False, transform=transforms.Compose([transforms.Grayscale(num_output_channels=IMG_CHANNELS), transforms.ToTensor()]), split='test')
    test_dataloader = torch.utils.data.DataLoader(LIDC_testset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    batch_images = next(iter(train_dataloader))

    print(f"Batch Size: {batch_images[0].shape[0]}, Number of Channels: {batch_images[0].shape[1]}, Image Size: {batch_images[0].shape[2]} x {batch_images[0].shape[3]} x {batch_images[0].shape[4]} (NCDHW)\n")
    print(f"Number of Characteristics: {len(batch_images[1])}")

    ###############################################################################################################
    ###################################### Initialize the model ###################################################
    ###############################################################################################################
    print("\n\n" + "#"*100 + "\n\n")
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Set the number of epochs (we'll keep this small for faster training times)
    epochs = 100

    if args.backbone == 'efficientNet3D':
        from efficientnet_pytorch_3d import EfficientNet3D #TODO: REMOVE IF NOT 3D
        model = EfficientNet3D.from_name("efficientnet-b0", in_channels=1, override_params={'num_classes': 1})
    else:
        if args.model not in MODEL_DICT:
            raise ValueError(f"Unsupported model name {args.model}")
        construct_Model = MODEL_DICT[args.model]
        if args.weights == 'None':
            args.weights = None
        # Create the model instance
        model = construct_Model(backbone_name=args.backbone, weights=args.weights)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    
    ###############################################################################################################
    ####################################### Training the model ####################################################
    ###############################################################################################################
    print("\n\n" + "#"*100 + "\n\n")
    # Initialize lists to store metrics over epochs
    all_train_metrics = []
    all_test_metrics = []

    # Train the model
    start_time = time.time()  # Record the start time of the entire training
    max_val_bacc = float(0)
    # min_val_loss = float('inf')
    for epoch in range(epochs):
        # Print header
        print("\n" + "-"*100 + f"\nEpoch: {epoch + 1}/{epochs},\n" + "-"*100)
        # Train and test the model batch by batch
        epoch_start = time.time()  # Start time of the current epoch

        # Training step
        train_metrics = train_step(data_loader=train_dataloader, 
                                    model=model, 
                                    optimizer=optimizer,
                                    device=device)
        all_train_metrics.append(train_metrics)  
        
        # Testing step
        test_metrics = test_step(data_loader=val_dataloader,
                                model=model,
                                device=device)
        all_test_metrics.append(test_metrics) 
        
        # Save the model if the val f1 has decreased
        if test_metrics['final_balanced_accuracy'] > max_val_bacc and test_metrics['final_balanced_accuracy'] > 0.70: # and epoch > 10:
            max_val_bacc = test_metrics['final_balanced_accuracy']
            save_model_in_chunks(model.state_dict(), best_model_path)

        epoch_end = time.time()  # End time of the current epoch
        print(f"\nEpoch {epoch + 1} completed in {epoch_end - epoch_start:.2f} seconds")  # Print the time taken for the epoch

    total_time = time.time() - start_time  # Total time for training
    print(f"Total training time: {total_time:.2f} seconds\n")  # Print the total training time

    save_metrics_to_csv(all_train_metrics, all_test_metrics, metrics_path)  # Save metrics to a CSV file
    plot_and_save_loss(all_train_metrics, all_test_metrics, plot_path)  # Plot and save the loss
    
    ###############################################################################################################
    ####################################### Evaluate the model ####################################################
    ###############################################################################################################
    print("\n\n" + "#"*100 + "\n\n")
    
    # Load the best model
    model.load_state_dict(load_model_from_chunks(best_model_path))
    
    # Evaluate the model on each slice
    test_metrics, test_confusion_matrix = evaluate_model(test_dataloader, model, device)
    print(f"Test Metrics:")
    print(test_metrics)
    print("Test Confusion Matrix:")
    print(test_confusion_matrix)    
    
    # Group slices by nodule and evaluate the model on each nodule
    # test set
    # LIDC_testset = LIDCEvaluationDataset(labels_file=labels_file, indeterminate=False, transform=transforms.Compose([transforms.Grayscale(num_output_channels=IMG_CHANNELS), transforms.ToTensor()]))
    # test_dataloader = torch.utils.data.DataLoader(LIDC_testset, batch_size=1, shuffle=False, num_workers=0) # Predict one nodule at a time
    # 
    # # Evaluate the model on the test set
    # test_metrics, test_confusion_matrix = evaluate_model_by_nodule(model, test_dataloader, device, mode="mean")
    # print(f"Test Metrics with Median Aggregation:")
    # print(test_metrics)
    # print("Test Confusion Matrix:")
    # print(test_confusion_matrix)
    # 
    # test_metrics, test_confusion_matrix = evaluate_model_by_nodule(model, test_dataloader, device, mode="median")
    # print(f"Test Metrics with Median Aggregation:")
    # print(test_metrics)
    # print("Test Confusion Matrix:")
    # print(test_confusion_matrix)
 
    # test_metrics, test_confusion_matrix = evaluate_model_by_nodule(model, test_dataloader, device, mode="gaussian", std_dev=0.6)
    # print(f"Test Metrics with Gaussian Aggregation and Standard Deviation of 0.6:")
    # print(test_metrics)
    # print("Test Confusion Matrix:")
    # print(test_confusion_matrix)
    # 
    # test_metrics, test_confusion_matrix = evaluate_model_by_nodule(model, test_dataloader, device, mode="gaussian", std_dev=0.8)
    # print(f"Test Metrics with Gaussian Aggregation and Standard Deviation of 0.8:")
    # print(test_metrics)
    # print("Test Confusion Matrix:")
    # print(test_confusion_matrix)
    # 
    # test_metrics, test_confusion_matrix = evaluate_model_by_nodule(model, test_dataloader, device, mode="gaussian", std_dev=1.0)
    # print(f"Test Metrics with Gaussian Aggregation and Standard Deviation of 1.0:")
    # print(test_metrics)
    # print("Test Confusion Matrix:")
    # print(test_confusion_matrix)
    # 
    # test_metrics, test_confusion_matrix = evaluate_model_by_nodule(model, test_dataloader, device, mode="gaussian", std_dev=1.2)
    # print(f"Test Metrics with Gaussian Aggregation and Standard Deviation of 1.2:")
    # print(test_metrics)
    # print("Test Confusion Matrix:")
    # print(test_confusion_matrix)
    # 
    # test_metrics, test_confusion_matrix = evaluate_model_by_nodule(model, test_dataloader, device, mode="gaussian", std_dev=1.4)
    # print(f"Test Metrics with Gaussian Aggregation and Standard Deviation of 1.4:")
    # print(test_metrics)
    # print("Test Confusion Matrix:")
    # print(test_confusion_matrix)
    # 
    # test_metrics, test_confusion_matrix = evaluate_model_by_nodule(model, test_dataloader, device, mode="gaussian", std_dev=1.6)
    # print(f"Test Metrics with Gaussian Aggregation and Standard Deviation of 1.6:")
    # print(test_metrics)
    # print("Test Confusion Matrix:")
    # print(test_confusion_matrix)
    # 
    # test_metrics, test_confusion_matrix = evaluate_model_by_nodule(model, test_dataloader, device, mode="gaussian", std_dev=1.8)
    # print(f"Test Metrics with Gaussian Aggregation and Standard Deviation of 1.8:")
    # print(test_metrics)
    # print("Test Confusion Matrix:")
    # print(test_confusion_matrix)
    # 
    # test_metrics, test_confusion_matrix = evaluate_model_by_nodule(model, test_dataloader, device, mode="gaussian", std_dev=2.0)
    # print(f"Test Metrics with Gaussian Aggregation and Standard Deviation of 2.0:")
    # print(test_metrics)
    # print("Test Confusion Matrix:")
    # print(test_confusion_matrix)
    
    # Save the test metrics to a CSV file
    df_test = pd.DataFrame([test_metrics])
    df_test.to_csv(test_metrics_path)

if __name__ == '__main__':
    main()
