from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import torch
import utils
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path

# Import modular functions (make sure all functions are correctly imported)
from going_modular.utils import (get_device, create_directory, get_project,
                                 download_files, construct_dataset_paths,
                                 download_videos_from_youtube)
from going_modular.coco_dataset import CustomCocoDataset
from going_modular.model_utils import get_model_instance_segmentation, load_classes_from_json
from going_modular.engine import train_model
from going_modular.process_video_check import process_video_check
from going_modular.transforms import get_transform, transform_image

app = FastAPI()

# Global variables (Consider storing and accessing these more securely and flexibly)
MODEL_PATH = Path('results/models/model_weights.pth')
 

# Ensure MODEL_PATH directory exists
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

def load_model(num_classes: int, model_path: Path = MODEL_PATH):
    model = get_model_instance_segmentation(num_classes, hidden_layer=256)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

@app.get("/")
async def root():
    return {"message": "Welcome to the API!"}

@app.post("/train")
async def train(api_key: str, workspace: str, project_name: str, project_folder_name: str, version: int, num_epochs: int = 10):
    # Set up device
    device = get_device()
    
    # Ensure data and model directories exist
    data_path = Path('results/data')
    model_path = Path('results/models')
    data_path.mkdir(parents=True, exist_ok=True)
    model_path.mkdir(parents=True, exist_ok=True)
    
    # Download and prepare the dataset
    dataset = get_project(api_key, workspace, project_name, version)
    train_annotation_path, valid_annotation_path, test_annotation_path, train_image_dir, valid_image_dir, test_image_dir = construct_dataset_paths(project_folder_name, version)
    
    # Load class names and set up the model
    CLASSES_JSON = Path(f'{project_folder_name}-{version}/test/_annotations.coco.json')
    classes = load_classes_from_json(CLASSES_JSON)
    num_classes = len(classes) + 1
    
    model = get_model_instance_segmentation(num_classes, hidden_layer=256)
    model.to(device)
    
    # Prepare datasets
    train_dataset = CustomCocoDataset(train_annotation_path, train_image_dir, transforms=get_transform(train=True))
    valid_dataset = CustomCocoDataset(valid_annotation_path, valid_image_dir, transforms=get_transform(train=False))
    
    # Set up data loaders
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=utils.collate_fn)
    valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=utils.collate_fn)
    
    # Train the model
    train_model(model, train_data_loader, valid_data_loader, device, num_epochs)
    
    # Save the trained model
    torch.save(model.state_dict(), MODEL_PATH)
    
    return {"message": "Model trained and saved successfully"}


@app.post("/predict")
async def predict(project_folder_name: str, version: int, file: UploadFile = File(...)):
    CLASSES_JSON = Path(f'{project_folder_name}-{version}/test/_annotations.coco.json')
    classes = load_classes_from_json(CLASSES_JSON)
    num_classes = len(classes) + 1
    model = load_model(num_classes)
    
    image_bytes = await file.read()
    tensor = transform_image(image_bytes)
    
    # Prediction logic
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_prob, top_catid = probabilities.topk(1, dim=1)
    
    predicted_class = classes[top_catid.item()]
    confidence = top_prob.item()
    
    return JSONResponse(content={"class": predicted_class, "confidence": confidence})

from fastapi import UploadFile, File
from pathlib import Path
import shutil

@app.post("/process_video")
async def process_video(project_folder_name: str, version: int, video_file: UploadFile = File(...)):
    # Define where to save the video temporarily
    temp_video_path = Path("temp_videos") / video_file.filename
    temp_video_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

    # Save the uploaded video to the temporary path
    with temp_video_path.open("wb") as buffer:
        shutil.copyfileobj(video_file.file, buffer)
    
    # Load model and classes for processing
    CLASSES_JSON = Path(f'{project_folder_name}-{version}/test/_annotations.coco.json')
    classes = load_classes_from_json(CLASSES_JSON)
    num_classes = len(classes) + 1
    model = load_model(num_classes)
    
    # Assuming process_video_check is adapted to return a meaningful result
    # For example, modifying process_video_check to accept a video path and return a dictionary of results
    results = process_video_check(str(temp_video_path), model, get_device(), classes, [('Basketball', 'Hoop')], threshold=0.6)

    # Optionally, delete the temporary video file after processing
    temp_video_path.unlink(missing_ok=True)
    
    return results

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)

#!uivcorn app:app --reload

#For looking at all the options**
#http://127.0.0.1:8000/docs
