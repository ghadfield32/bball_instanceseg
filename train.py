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

print(torch.__version__)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

def main(args):
    data_path = Path(args.data_path)
    model_path = Path(args.model_path)
    data_path.mkdir(parents=True, exist_ok=True) #redundant
    model_path.mkdir(parents=True, exist_ok=True) #redundant

    dataset = get_project(args.api_key, args.workspace, args.project_name, args.version)

    classes = load_classes_from_json(f'{args.project_folder_name}-{args.version}/train/_annotations.coco.json')
    print("Classes loaded:", classes)

    num_classes = len(classes) + 1
    device = get_device()

    train_annotation_path, valid_annotation_path, test_annotation_path, train_image_dir, valid_image_dir, test_image_dir = construct_dataset_paths(args.project_folder_name, args.version)

    train_dataset = CustomCocoDataset(train_annotation_path, train_image_dir, transforms=get_transform(train=True))
    valid_dataset = CustomCocoDataset(valid_annotation_path, valid_image_dir, transforms=get_transform(train=False))
    test_dataset = CustomCocoDataset(test_annotation_path, test_image_dir)

    model = get_model_instance_segmentation(num_classes, hidden_layer=args.hidden_layer)
    model.to(device)

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=utils.collate_fn)
    valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=utils.collate_fn)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=utils.collate_fn)

    if args.mode == 'train':
        train_model(model, train_data_loader, valid_data_loader, device, args.num_epochs, lr=args.lr)
        model_file_path = model_path / 'model_weights.pth'
        try:
            torch.save(model.state_dict(), str(model_file_path))
            print("Model saved at:", model_file_path)
        except Exception as e:
            print(f"Error saving model: {e}")

    elif args.mode == 'process_video':
        model_file_path = model_path / 'model_weights.pth'
        if not model_file_path.exists():
            print("Model weights file not found. Please train the model first.")
            return

        # Download the video
        successful_downloads, failed_downloads = download_videos_from_youtube([args.video_url], str(data_path))
        print(f"Successfully downloaded {len(successful_downloads)} videos.")
        print(f"Failed to download {len(failed_downloads)} videos.")

        if successful_downloads:
            # Use the video filename from the command-line argument
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model for object detection or process video")
    parser.add_argument('--api_key', type=str, default="your_roboflow_api_key", help='API key for Roboflow')
    parser.add_argument('--workspace', type=str, default="your_roboflow_workspace", help='Workspace name in Roboflow')
    parser.add_argument('--project_name', type=str, default="your_roboflow_project", help='Project name in Roboflow')
    parser.add_argument('--project_folder_name', type=str, default="your_roboflow_project_folder_name", help='Project folder name in Roboflow')
    parser.add_argument('--version', type=int, default=1, help='Version of the dataset in Roboflow')
    parser.add_argument('--hidden_layer', type=int, default=256, help='Hidden layer size for the MaskRCNN predictor')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train the model')
    parser.add_argument('--video_url', type=str, default="https://www.youtube.com/watch?v=example_video_id", help='URL of the video to process')
    parser.add_argument('--video_name', type=str, default="your_youtube_video_name", help='Name of the video file (with extension) to process')
    parser.add_argument('--threshold', type=float, default=0.6, help='Detection threshold for process_video')
    parser.add_argument('--display_video', type=bool, default=True, help='Whether to display the video during processing')
    parser.add_argument('--data_path', type=str, default='results/data', help='Path to save downloaded data')
    parser.add_argument('--model_path', type=str, default='results/models', help='Path to save model weights')
    parser.add_argument('--mode', type=str, choices=['train', 'process_video'], default='train', help='Mode of operation: train or process_video')

    args = parser.parse_args()
    main(args)
