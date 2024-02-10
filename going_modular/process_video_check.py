
import cv2
import torch
import numpy as np
from torchvision.transforms import v2 as T
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks


def intersects(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1.tolist()
    x2_min, y2_min, x2_max, y2_max = box2.tolist()
    return (x1_min < x2_max and x1_max > x2_min and y1_min < y2_max and y1_max > y2_min)

def process_video_check(video_path, model, device, classes, classes_to_track=None, threshold=0.5, check_intersections=False):
    model.eval()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Error opening video file")
        return

    score = 0  # Initialize the score counter

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_tensor = T.ToTensor()(frame).unsqueeze_(0).to(device)

        with torch.no_grad():
            prediction = model(frame_tensor)[0]

        pred_scores = prediction['scores']
        pred_boxes = prediction['boxes']
        pred_labels = prediction['labels']
        pred_masks = prediction['masks']

        keep = pred_scores > threshold
        pred_boxes = pred_boxes[keep]
        pred_labels = pred_labels[keep]
        pred_masks = pred_masks[keep]

        #print(f"Original Frame Size: {frame.shape}")
        #print(f"Pred Boxes before drawing: {pred_boxes}")

        if not keep.any():
            continue  # Skip this frame if no detections are kept

        # Convert numeric labels to class names
        pred_class_names = [classes[label.item()] for label in pred_labels]

        if check_intersections == True and classes_to_track:
            # Perform intersection checks only if enabled and classes_to_track is specified
            for class_pair in classes_to_track:
                class1_boxes = pred_boxes[[name == class_pair[0] for name in pred_class_names]]
                class2_boxes = pred_boxes[[name == class_pair[1] for name in pred_class_names]]

                for box1 in class1_boxes:
                    for box2 in class2_boxes:
                        if intersects(box1, box2):
                            score += 1
                            print(f"Intersection detected between {class_pair[0]} and {class_pair[1]}, Score:", score)

        # Frame Tensor Conversion for Drawing
        frame_tensor = (255.0 * (frame_tensor - frame_tensor.min()) / (frame_tensor.max() - frame_tensor.min())).to(torch.uint8)
        frame_tensor = frame_tensor.squeeze().to(torch.uint8)

        # Draw bounding boxes and segmentation masks
        output_image = draw_bounding_boxes(frame_tensor, pred_boxes, labels=pred_class_names, colors="red")
        output_image = draw_segmentation_masks(output_image, (pred_masks > 0.7).squeeze(1), alpha=0.5, colors="blue")

        # Convert output image for displaying
        output_image = output_image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        output_image = np.clip(output_image, 0, 255)  # Ensure values are within 0-255
        if check_intersections == True and classes_to_track:
            # Draw score text
            cv2.putText(output_image, f'Score: {score}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('Frame', output_image)
        
        # Just before cv2.imshow
        #print(f"Output Image Size: {output_image.shape}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
# process_video_check(video_path, model, device, classes, [('ball', 'rim')], threshold=0.5)
