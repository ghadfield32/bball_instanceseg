import matplotlib.pyplot as plt
import torchvision.transforms.functional as F  # Add this import

# New function to visualize transformations
def visualize_transformation(dataset, idx):
    img, target = dataset[idx]
    transformed_img, transformed_target = dataset.transforms(img, target)
    original_img = F.to_pil_image(img)
    transformed_img = F.to_pil_image(transformed_img)

    plt.figure(figsize=(24, 6))
    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    for box in target["boxes"]:
        x_min, y_min, x_max, y_max = box.tolist()
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)
        #print(x_min, y_min, x_max, y_max)
    plt.title(f"Original Image - ID: {idx}")

    # Transformed Image
    plt.subplot(1, 2, 2)
    plt.imshow(transformed_img)
    for box in transformed_target["boxes"]:
        x_min, y_min, x_max, y_max = box.tolist()
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='b', facecolor='none')
        plt.gca().add_patch(rect)
        #print(x_min, y_min, x_max, y_max)
    plt.title(f"Transformed Image - ID: {idx}")
    plt.show()



def visualize_bbox(dataset, idx):
    img, target = dataset[idx]
    original_img = F.to_pil_image(img)

    plt.figure(figsize=(12, 6))
    plt.imshow(original_img)

    for box in target["boxes"]:  # Access the boxes directly
        x_min, y_min, x_max, y_max = box.tolist()
        # Debug print
        print(f"Visualizing BBox - xmin: {x_min}, ymin: {y_min}, xmax: {x_max}, ymax: {y_max}")
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)

    plt.title(f"Image with Bounding Boxes - ID: {idx}")
    plt.show()
