import cv2
import torch
from collections import deque
from torchvision.transforms import v2 as T
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

def process_video(video_path, model, device, classes, threshold=0.5, ball_class_idx=1, rim_class_idx=3, display=True):
    model.eval()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video file")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Buffer to store frames
    buffer_length = 10  # 5 frames before and 5 frames after
    frame_buffer = deque(maxlen=buffer_length)

    intersection_detected = False
    saved_clip = False
    output_video_path = "data/intersection_clip.mp4"
    video_writer = None

    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        frame_tensor = T.ToTensor()(frame).unsqueeze_(0).to(device)

        # Model inference
        with torch.no_grad():
            prediction = model(frame_tensor)[0]

        # Post-processing
        pred_scores = prediction['scores']
        pred_boxes = prediction['boxes']
        pred_labels = prediction['labels']
        pred_masks = prediction['masks']

        # Filter predictions based on threshold
        keep = pred_scores > threshold
        pred_boxes = pred_boxes[keep]
        pred_labels = pred_labels[keep]
        pred_masks = pred_masks[keep]

        # Ensure labels are strings
        label_texts = [f'Label: {classes[label.item()]}, Score: {score:.3f}' for label, score in zip(pred_labels, pred_scores[keep])]

        # Convert to uint8 for drawing bounding boxes and masks
        frame_tensor_uint8 = (255.0 * frame_tensor.squeeze()).to(torch.uint8)
        output_image = draw_bounding_boxes(frame_tensor_uint8, pred_boxes, labels=label_texts, colors="red")
        output_image = draw_segmentation_masks(output_image, pred_masks > 0.5, alpha=0.5, colors="blue")

        # Convert tensor back to BGR image for display with OpenCV
        output_image = output_image.permute(1, 2, 0).cpu().numpy()
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

        # Check for intersection
        intersection = intersects(pred_boxes, pred_labels, ball_class_idx, rim_class_idx)
        if intersection and not intersection_detected:
            intersection_detected = True
            # Initialize video writer
            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
            # Write buffered frames
            for buffered_frame in frame_buffer:
                video_writer.write(buffered_frame)

        if intersection_detected and not saved_clip:
            video_writer.write(frame)
            if len(frame_buffer) == buffer_length - 1:  # Saved all frames after intersection
                saved_clip = True
                video_writer.release()

        if display:
            cv2.imshow('Frame', output_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_buffer.append(frame)

    cap.release()
    cv2.destroyAllWindows()
    if video_writer is not None and not saved_clip:
        video_writer.release()

def intersects(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1.tolist()
    x2_min, y2_min, x2_max, y2_max = box2.tolist()
    return (x1_min < x2_max and x1_max > x2_min and y1_min < y2_max and y1_max > y2_min)


#threshold = 0.5  # Adjust as needed

# Usage
#video_path = video_path  # Replace with your video path
#process_video(video_path, model, device, classes, threshold=threshold, ball_class_idx=1, rim_class_idx=3, display=display_video)
