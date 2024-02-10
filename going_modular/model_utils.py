import json
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import utils
import shutil
import os

def load_classes_from_json(file_path):
    """
    Loads the class names and their corresponding IDs from a COCO format JSON file.

    Args:
    file_path (str): Path to the JSON file.

    Returns:
    dict: A dictionary where keys are class IDs and values are class names.
    """
    with open(file_path) as f:
        data = json.load(f)
    categories = data['categories']
    classes = {category['id']: category['name'] for category in categories}
    return classes



# Usage example:
#classes = load_classes_from_json('basketball_child-6/test/_annotations.coco.json')
#print(classes)

# model_utils.py
def get_model_instance_segmentation(num_classes, hidden_layer=256):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model




def upload_to_huggingface(model_directory, model_id):
    """Upload the model to Hugging Face Hub."""
    hf_api = HfApi()
    username = hf_api.whoami()['name']
    repo_name = f"{username}/{model_id}"
    repo_url = hf_api.create_repo(repo_name, exist_ok=True, private=False)

    repo = Repository(local_dir=model_directory, clone_from=repo_url, use_auth_token=True)
    repo.lfs_track(["*.bin", "*.pth", "*.ckpt"])  # Track large model files with Git LFS
    repo.git_add()
    repo.git_commit("Initial commit of the model")
    try:
        repo.git_push()
        print(f"Model successfully uploaded to: {repo_url}")
    except Exception as e:
        print(f"Failed to upload model to Hugging Face: {e}")
        
