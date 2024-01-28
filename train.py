import argparse
import os
from going_modular.utils import (get_device, create_directory, get_project,
                                 download_files, construct_dataset_paths,
                                 download_videos_from_youtube)
from going_modular.coco_dataset import CustomCocoDataset
from going_modular.model_utils import (get_model_instance_segmentation,
                                       load_classes_from_json)
from going_modular.engine import train_model
from going_modular.transforms import get_transform
from going_modular.process_video_check import process_video_check
import utils
import torch
from torch import nn
from pathlib import Path

# Note: this notebook requires torch >= 1.10.0
print(torch.__version__)

# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
device

def main(args):
    # Create directories for data and models
    data_path = Path(args.data_path)
    model_path = Path(args.model_path)
    data_path.mkdir(parents=True, exist_ok=True)
    model_path.mkdir(parents=True, exist_ok=True)

    # Get the project dataset
    dataset = get_project(args.api_key, args.workspace, args.project_name, args.version)

    # Load classes from json
    classes = load_classes_from_json(f'{args.project_folder_name}-{args.version}/test/_annotations.coco.json')
    print(classes)

    # Set the number of classes
    num_classes = len(classes) + 1

    # Initialize device
    device = get_device()

    # Construct dataset paths
    train_annotation_path, valid_annotation_path, test_annotation_path, train_image_dir, valid_image_dir, test_image_dir = construct_dataset_paths(args.project_folder_name, args.version)

    # Create datasets with transforms
    train_dataset = CustomCocoDataset(train_annotation_path, train_image_dir, transforms=get_transform(train=True))
    valid_dataset = CustomCocoDataset(valid_annotation_path, valid_image_dir, transforms=get_transform(train=False))
    test_dataset = CustomCocoDataset(test_annotation_path, test_image_dir)

    # Load model
    model = get_model_instance_segmentation(num_classes, 
                                            hidden_layer=args.hidden_layer)
    model.to(device)

    # Data Loaders
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=utils.collate_fn)
    valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=utils.collate_fn)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=utils.collate_fn)

    # Train the model
    train_model(model, train_data_loader, valid_data_loader, device, args.num_epochs, lr=args.lr)

    # Save model after training
    model_file_path = model_path / 'model_weights.pth'
    try:
        torch.save(model.state_dict(), str(model_file_path))
        print("Model saved at:", model_file_path)
    except Exception as e:
        print(f"Error saving model: {e}")

    # Download and process video
    try:
        successful_downloads, failed_downloads = download_videos_from_youtube([args.video_url], args.data_path)
        print(f"Successfully downloaded {len(successful_downloads)} videos.")
        print(f"Failed to download {len(failed_downloads)} videos.")

        if successful_downloads:
            # Construct video path
            video_filename = args.video_name
            if not video_filename.endswith('.mp4'):
                video_filename += '.mp4'
            video_path = data_path / video_filename
            print(f"Video path: {video_path}")

            # Load model state for video processing
            model.load_state_dict(torch.load(str(model_file_path), map_location=device))
            model.to(device)

            # Process video
            process_video_check(video_path, model, device, classes, [('Basketball', 'Hoop')], threshold=args.threshold)
        else:
            print("No videos were downloaded.")

    except Exception as e:
        print(f"Error in video downloading or processing: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model for object detection")
    # Define default values
    default_api_key = "your_roboflow_api_key"
    default_workspace = "your_roboflow_workspace"
    default_project_name = "your_roboflow_project"
    default_project_folder_name = "your_roboflow_project_folder_name"
    default_version = 1
    default_hidden_layer = 256
    default_lr = 0.005
    default_num_epochs = 10
    default_video_url = "https://www.youtube.com/watch?v=example_video_id"
    default_video_name = "your_youtube_video_name"
    default_confidence_threshold = 0.6
    default_display_video = True
    default_data_path = 'results/data'
    default_model_path = 'results/models'

    # Setup argparse with defaults
    parser.add_argument('--api_key', type=str, default=default_api_key, help='API key for Roboflow')
    parser.add_argument('--workspace', type=str, default=default_workspace, help='Workspace name in Roboflow')
    parser.add_argument('--project_name', type=str, default=default_project_name, help='Project name in Roboflow')
    parser.add_argument('--project_folder_name', type=str, default=default_project_folder_name, help='Project folder name in Roboflow')
    parser.add_argument('--version', type=int, default=default_version, help='Version of the dataset in Roboflow')
    parser.add_argument('--hidden_layer', type=int, default=default_hidden_layer, help='Hidden layer size for the MaskRCNN predictor')
    parser.add_argument('--lr', type=float, default=default_lr, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=default_num_epochs, help='Number of epochs to train the model')
    parser.add_argument('--video_url', type=str, default=default_video_url, help='URL of the video to process')
    parser.add_argument('--video_name', type=str, default=default_video_name, help='Name of the video file (with extension) to process')
    parser.add_argument('--threshold', type=float, default=default_confidence_threshold, help='Detection threshold for process_video')
    parser.add_argument('--display_video', type=bool, default=default_display_video, help='Whether to display the video during processing')
    parser.add_argument('--data_path', type=str, default=default_data_path, help='Path to save downloaded data')
    parser.add_argument('--model_path', type=str, default=default_model_path, help='Path to save model weights')

    args = parser.parse_args()
    main(args)
