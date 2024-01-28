import json
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def load_classes_from_json(file_path):
    """
    Loads the class names and their corresponding IDs from a COCO format JSON file.

    Args:
    file_path (str): Path to the JSON file.

    Returns:
    dict: A dictionary where keys are class IDs and values are class names.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Extracting classes and their IDs
    classes = {category['id']: category['name'] for category in data['categories']}
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

