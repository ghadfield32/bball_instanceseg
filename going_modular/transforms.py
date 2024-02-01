import torch  # Add this import statement
from torchvision.transforms import v2 as T
from torchvision.transforms import Compose, RandomHorizontalFlip, ToTensor, ConvertImageDtype
from torchvision import transforms
from PIL import Image
from io import BytesIO

def get_transform(train):
    transforms = []
    #if train:
    #    transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

def transform_image(image_bytes):
    """
    Transforms image bytes into a tensor with the correct format for the model.
    
    Args:
    image_bytes (bytes): The image in bytes format, as uploaded by the user.
    
    Returns:
    torch.Tensor: The transformed image as a tensor.
    """
    # Define the transformations
    my_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to the size required by your model
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Standard normalization for pre-trained models
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Load the image from bytes and apply transformations
    image = Image.open(BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)  # Add a batch dimension
