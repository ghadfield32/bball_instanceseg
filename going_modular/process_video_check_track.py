
import cv2
import torch
import numpy as np
from torchvision.transforms import v2 as T
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker

# Define ByteTrack arguments (customize as needed)
class ByteTrackArgument:
    track_thresh = 0.5
    track_buffer = 30
    match_thresh = 0.8
    aspect_ratio_thresh = 1.6
    min_box_area = 10

def intersects(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1.tolist()
    x2_min, y2_min, x2_max, y2_max = box2.tolist()
    return (x1_min < x2_max and x1_max > x2_min and y1_min < y2_max and y1_max > y2_min)

def process_video_check_track(video_path, model, device, classes, classes_to_track, threshold=0.5):
    model.eval()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Error opening video file")
        return

    # Initialize ByteTrack
    tracker = BYTETracker(ByteTrackArgument)

    score_counter = 0  # Initialize a separate score counter

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Debug: Check the type and shape of the frame
        print(f"Frame type: {type(frame)}, Frame shape: {frame.shape}")

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

        if not keep.any():
            continue  # Skip this frame if no detections are kept

        # Convert numeric labels to class names
        pred_class_names = [classes[label.item()] for label in pred_labels]

        # Iterate through each pair of classes to track
        for class_pair in classes_to_track:
            class1_boxes = pred_boxes[[name == class_pair[0] for name in pred_class_names]]
            class2_boxes = pred_boxes[[name == class_pair[1] for name in pred_class_names]]

            # Check for intersections and update score counter
            for box1 in class1_boxes:
                for box2 in class2_boxes:
                    if intersects(box1, box2):
                        score_counter += 1
                        print(f"Intersection detected between {class_pair[0]} and {class_pair[1]}, Score:", score_counter)

        # Frame Tensor Conversion for Drawing
        frame_tensor = (255.0 * (frame_tensor - frame_tensor.min()) / (frame_tensor.max() - frame_tensor.min())).to(torch.uint8)
        frame_tensor = frame_tensor.squeeze().to(torch.uint8)

        # Draw bounding boxes and segmentation masks
        output_image = draw_bounding_boxes(frame_tensor, pred_boxes, labels=pred_class_names, colors="red")
        output_image = draw_segmentation_masks(output_image, (pred_masks > 0.7).squeeze(1), alpha=0.5, colors="blue")

        # Convert output image for displaying
        output_image = output_image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        output_image = np.clip(output_image, 0, 255)  # Ensure values are within 0-255

        # Draw score text
        cv2.putText(output_image, f'Score: {score_counter}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Prepare detections for ByteTrack
        dets = []
        for box, det_score in zip(pred_boxes, pred_scores):
            if det_score > threshold:
                x1, y1, x2, y2 = box.tolist()
                current_det = [x1, y1, x2, y2, det_score.item()]
                dets.append(current_det)

        # Frame information for BYTETracker
        img_info = {"height": frame.shape[0], "width": frame.shape[1]}
        img_size = [frame.shape[1], frame.shape[0]]

        # Update ByteTrack
        online_targets = tracker.update(dets, img_info, img_size)

        # Process tracking results and draw on frame
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            # Draw tracking ID and box (customize as needed)
            cv2.rectangle(frame, (int(tlwh[0]), int(tlwh[1])), (int(tlwh[0]+tlwh[2]), int(tlwh[1]+tlwh[3])), (0,255,0), 2)
            cv2.putText(frame, str(tid), (int(tlwh[0]), int(tlwh[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Example usage
# process_video_check_track(video_path, model, device, classes, [('ball', 'rim')], threshold=0.5)
