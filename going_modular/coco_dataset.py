
import json
import os
import numpy as np
from PIL import Image, ImageDraw
import torch
from torchvision.transforms.v2 import functional as F
from torchvision import tv_tensors

class CustomCocoDataset(torch.utils.data.Dataset):
    def __init__(self, annotation_path, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms

        with open(annotation_path) as f:
            self.annotations = json.load(f)

        # Filter out images without annotations
        annotated_images = []
        for img in self.annotations['images']:
            image_id = img['id']
            anns = [ann for ann in self.annotations['annotations'] if ann['image_id'] == image_id]
            if len(anns) > 0:
                annotated_images.append(img)

        self.image_ids = [img['id'] for img in annotated_images]

        # Update the self.annotations['images'] to include only annotated images
        self.annotations['images'] = annotated_images
        
        #print("Number of images:", len(self.annotations['images']))
        #print("Sample image entry:", self.annotations['images'][0])

    def __len__(self):
        return len(self.annotations['images'])

    def __getitem__(self, idx):
        img_info = self.annotations['images'][idx]
        image_id = img_info['id']
        
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")
        img_tensor = F.to_tensor(img)
        #print("Image size (PIL):", img.size)
        #print("Image shape (tensor):", img_tensor.shape)

        anns = [ann for ann in self.annotations['annotations'] if ann['image_id'] == image_id]
        #print("Number of annotations for this image:", len(anns))

        boxes = [ann['bbox'] for ann in anns]  # bbox format: [x_min, y_min, width, height]
        # Convert from XYWH to XYXY format
        boxes = [[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in boxes]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = [ann['category_id'] for ann in anns]
        labels = torch.as_tensor(labels, dtype=torch.int64)
        #print("Boxes shape:", boxes.shape)
        #print("Labels:", labels)
        # Debug print
        #print(f"Boxes shape for image {idx}: {boxes.shape}")

        masks = []
        for ann in anns:
            if 'segmentation' in ann and isinstance(ann['segmentation'], list):
                for seg in ann['segmentation']:
                    mask_img = Image.new('L', (img_info['width'], img_info['height']), 0)
                    ImageDraw.Draw(mask_img).polygon(seg, outline=1, fill=1)
                    mask = np.array(mask_img)
                    masks.append(mask)
        masks = torch.as_tensor(np.array(masks), dtype=torch.uint8) if masks else torch.zeros((0, img_info['height'], img_info['width']), dtype=torch.uint8)
        #print("Masks shape:", masks.shape)

        areas = [ann['area'] for ann in anns]
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = [ann['iscrowd'] for ann in anns]
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        # Convert masks to Mask format
        masks = tv_tensors.Mask(masks)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id  # Changed to integer
        target["area"] = areas
        target["iscrowd"] = iscrowd

        #print("Target:", target)

        if self.transforms is not None:
            img_tensor, target = self.transforms(img_tensor, target)

        return img_tensor, target



