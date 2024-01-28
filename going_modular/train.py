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
import utils
import torch
from torch import nn

# Note: this notebook requires torch >= 1.10.0
print(torch.__version__)

# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
device

def main(args):
    # Create directories for data and models
    create_directory(args.data_path)
    create_directory(args.model_path)

    # Download required files from torchvision
    urls = [
        "https://raw.githubusercontent.com/pytorch/vision/main/references/detection/engine.py",
        "https://raw.githubusercontent.com/pytorch/vision/main/references/detection/utils.py",
        "https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_utils.py",
        "https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_eval.py",
        "https://raw.githubusercontent.com/pytorch/vision/main/references/detection/transforms.py"
    ]
    download_files(urls)

    # Get the project dataset
    dataset = get_project(args.api_key, args.workspace, args.project_name, args.version)

    # Load classes from json
    classes = load_classes_from_json(f'{args.project_name}-{args.version}/test/_annotations.coco.json')
    print(classes)

    # Set the number of classes
    num_classes = len(classes) + 1

    # Initialize device
    device = get_device()

    # Construct dataset paths
    train_annotation_path, valid_annotation_path, test_annotation_path, train_image_dir, valid_image_dir, test_image_dir = construct_dataset_paths(args.project_name, args.version)

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
    model_file_path = os.path.join(args.model_path, 'model_weights.pth')
    try:
        torch.save(model.state_dict(), model_file_path)
        print("Model saved at:", model_file_path)
    except Exception as e:
        print(f"Error saving model: {e}")

    # Download and process video
    try:
        successful_downloads, failed_downloads = download_videos_from_youtube([args.video_url], args.data_path)
        print(f"Successfully downloaded {len(successful_downloads)} videos.")
        print(f"Failed to download {len(failed_downloads)} videos.")
    except Exception as e:
        print(f"Error downloading video: {e}")

    video_filename = "downloaded_inference_video.mp4"
    video_path = os.path.join(args.data_path, video_filename)

    # Load model state for video processing
    try:
        model.load_state_dict(torch.load(model_file_path, map_location=device))
        model.to(device)
        print("Model loaded from:", model_file_path)
    except Exception as e:
        print(f"Error loading model: {e}")

    # Process video
    try:
        process_video(video_path, model, device, classes, threshold=args.threshold, ball_class_idx=1, rim_class_idx=3, display=args.display_video)
    except Exception as e:
        print(f"Error processing video: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model for object detection")
    # Define default values
    default_api_key = "your_roboflow_api_key"
    default_workspace = "your_roboflow_workspace"
    default_project_name = "your_roboflow_project"
    default_version = 1
    default_hidden_layer = 256
    default_lr = 0.005
    default_num_epochs = 10
    default_video_url = "https://www.youtube.com/watch?v=example_video_id"
    default_confidence_threshold = 0.6
    default_display_video = True
    default_data_path = 'results/data'
    default_model_path = 'results/models'

    # Setup argparse with defaults
    parser.add_argument('--api_key', type=str, default=default_api_key, help='API key for Roboflow')
    parser.add_argument('--workspace', type=str, default=default_workspace, help='Workspace name in Roboflow')
    parser.add_argument('--project_name', type=str, default=default_project_name, help='Project name in Roboflow')
    parser.add_argument('--version', type=int, default=default_version, help='Version of the dataset in Roboflow')
    parser.add_argument('--hidden_layer', type=int, default=default_hidden_layer, help='Hidden layer size for the MaskRCNN predictor')
    parser.add_argument('--lr', type=float, default=default_lr, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=default_num_epochs, help='Number of epochs to train the model')
    parser.add_argument('--video_url', type=str, default=default_video_url, help='URL of the video to process')
    parser.add_argument('--threshold', type=float, default=default_confidence_threshold, help='Detection threshold for process_video')
    parser.add_argument('--display_video', type=bool, default=default_display_video, help='Whether to display the video during processing')
    parser.add_argument('--data_path', type=str, default=default_data_path, help='Path to save downloaded data')
    parser.add_argument('--model_path', type=str, default=default_model_path, help='Path to save model weights')

    args = parser.parse_args()
    main(args)


#example usage
# python train.py --api_key YOUR_API_KEY \
#                 --workspace YOUR_WORKSPACE \
#                 --project_name YOUR_PROJECT_NAME \
#                 --version YOUR_VERSION \
#                 --hidden_layer 256 \
#                 --lr 0.005 \
#                 --num_epochs 10 \
#                 --threshold 0.6 \
#                 --video_url "https://www.youtube.com/watch?v=VIDEO_ID"


