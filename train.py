import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import random_split, DataLoader
from going_modular.utils import get_device, create_directory, get_project, download_videos_from_youtube
from going_modular.coco_dataset import CustomCocoDataset
from going_modular.model_utils import get_model_instance_segmentation
from going_modular.engine import train_model
from going_modular.transforms import get_transform
from going_modular.process_video_check import process_video_check
import utils

def load_classes_from_json(annotation_path):
    with open(annotation_path) as f:
        data = json.load(f)
    categories = data['categories']
    classes = {category['id']: category['name'] for category in categories}
    return classes

def split_dataset(dataset, split_ratio=0.8):
    total_size = len(dataset)
    train_size = int(total_size * split_ratio)
    valid_size = total_size - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(42))
    return train_dataset, valid_dataset

def main(args):
    # Create model path directory
    model_path = Path(args.model_path)
    create_directory(model_path)

    # Check if dataset is already downloaded
    project_folder = Path.cwd() / f'{args.project_folder_name}-{args.version}'
    print(f"Checking dataset at: {project_folder.absolute()}")
    
    if args.mode == 'train':
        if not project_folder.exists():
            print("Downloading dataset...")
            dataset = get_project(args.api_key, args.workspace, args.project_name, args.version)
        else:
            print("Project data already downloaded. Skipping download...")

        # Load classes from annotation file
        classes_path = project_folder / 'train' / '_annotations.coco.json'
        if not classes_path.exists():
            print(f"Error: Annotation file for the train dataset not found at {classes_path.absolute()}.")
            return
        classes = load_classes_from_json(classes_path)
        print("Classes loaded:", classes)

        # Initialize model
        num_classes = len(classes) + 1
        device = get_device()
        model = get_model_instance_segmentation(num_classes, hidden_layer=args.hidden_layer)
        model.to(device)

        # Load datasets
        datasets = {}
        data_loaders = {}
        for dtype in ['train', 'valid', 'test']:
            ann_path = project_folder / dtype / '_annotations.coco.json'
            img_dir = project_folder / dtype
            if ann_path.exists() and img_dir.exists():
                datasets[dtype] = CustomCocoDataset(str(ann_path), str(img_dir), transforms=get_transform(train=dtype=='train'))
                batch_size = 2 if dtype == 'train' else 1
                data_loaders[dtype] = DataLoader(datasets[dtype], batch_size=batch_size, shuffle=dtype=='train', num_workers=0, collate_fn=utils.collate_fn)
                print(f"{dtype.capitalize()} dataset loaded.")
            else:
                print(f"{dtype.capitalize()} dataset not found or incomplete. Skipping.")

        if 'train' in datasets and 'valid' not in datasets:
            print("Splitting dataset into train and valid...")
            train_dataset, valid_dataset = split_dataset(datasets['train'])
            data_loaders['train'] = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=utils.collate_fn)
            data_loaders['valid'] = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=utils.collate_fn)
            
        if 'train' in data_loaders and 'valid' in data_loaders:
            print("Starting training process...")
            train_model(model, data_loaders['train'], data_loaders.get('valid'), device, args.num_epochs, lr=args.lr)
            model_file_path = model_path / 'model_weights.pth'
            torch.save(model.state_dict(), model_file_path)
            print(f"Model saved at: {model_file_path}")
        else:
            print("Training failed. No valid data loaders available.")

    # Process video if mode is 'process_video'
    if args.mode == 'process_video':
        video_filename = args.video_name if args.video_name.endswith('.mp4') else f"{args.video_name}.mp4"
        video_path = Path(args.data_path) / video_filename
        if not video_path.exists():
            print("Downloading video...")
            download_videos_from_youtube([args.video_url], str(Path(args.data_path)))
        
        if video_path.exists():
            model_file_path = model_path / 'model_weights.pth'
            if model_file_path.exists():
                model.load_state_dict(torch.load(str(model_file_path), map_location=device))
                process_video_check(video_path, model, device, classes, [('Basketball', 'Hoop')], threshold=args.threshold)
            else:
                print("Model weights file not found. Please train the model first.")

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
