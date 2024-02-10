import cv2
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor, convert_image_dtype
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
import torchvision.transforms.functional as F
from bytetracker.kalman_filter import KalmanFilter
from torchvision.ops import box_convert

# Utility Functions
def tlwh_to_xyah(tlwh):
    x, y, w, h = tlwh
    center_x = x + w / 2
    center_y = y + h / 2
    aspect_ratio = w / h
    return np.array([center_x, center_y, aspect_ratio, h])

def xyah_to_tlwh(xyah):
    """
    Convert the state vector (including velocity or other components)
    back to the top-left x, y, width, height (tlwh) bounding box format.
    """
    # Ensure xyah only contains [center_x, center_y, aspect_ratio, height]
    if len(xyah) > 4:
        xyah = xyah[:4]  # Consider only the first 4 components for conversion
    center_x, center_y, aspect_ratio, h = xyah
    w = aspect_ratio * h
    tlwh = np.array([center_x - w / 2, center_y - h / 2, w, h])
    return tlwh

# Track Class
class Track:
    _id_count = 1
    
    def __init__(self, tlwh, score):
        self.kalman_filter = KalmanFilter()
        self.xyah = tlwh_to_xyah(tlwh)
        self.state, self.covariance = self.kalman_filter.initiate(self.xyah)
        self.track_id = Track._id_count
        Track._id_count += 1
        self.score = score
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

    def predict(self):
        self.state, self.covariance = self.kalman_filter.predict(self.state, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, tlwh):
        self.xyah = tlwh_to_xyah(tlwh)
        self.state, self.covariance = self.kalman_filter.update(self.state, self.covariance, self.xyah)
        self.hits += 1
        self.time_since_update = 0

    def get_tlwh(self):
        return xyah_to_tlwh(self.state[:4])

def iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1+w1, x2+w2), min(y1+h1, y2+h2)
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
    box1_area, box2_area = w1*h1, w2*h2
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def adjust_boxes(boxes, scale_factor=(1.0, 1.0)):
    """Adjust bounding boxes by a scale factor."""
    adjusted_boxes = []
    for box in boxes:
        # Assuming box is in tlwh format
        x, y, w, h = box
        adjusted_box = [x * scale_factor[0], y * scale_factor[1], (x + w) * scale_factor[0], (y + h) * scale_factor[1]]
        adjusted_boxes.append(adjusted_box)
    return adjusted_boxes

def process_video_check_track(video_path, model, device, threshold=0.5):
    model.eval()
    cap = cv2.VideoCapture(str(video_path))
    tracks = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        original_width, original_height = frame.shape[1], frame.shape[0]
        frame_tensor = to_tensor(frame).unsqueeze_(0).to(device)
        with torch.no_grad():
            prediction = model(frame_tensor)[0]

        keep = prediction['scores'] > threshold
        boxes = prediction['boxes'][keep].cpu()
        scores = prediction['scores'][keep].cpu()

        # Ensure the loop for processing detections and adjusting boxes is correctly scoped
        for i, box in enumerate(boxes):
            det = box.numpy()  # Convert to numpy array
            score = scores[i].item()

            # Scale factor for adjusting boxes to the original frame size
            scale_factor = (original_width / frame_tensor.shape[3], original_height / frame_tensor.shape[2])
            adjusted_box = adjust_boxes([det], scale_factor)[0]
            adjusted_box_tensor = torch.tensor([adjusted_box], dtype=torch.float32)
            adjusted_tlbr = box_convert(adjusted_box_tensor, in_fmt='xywh', out_fmt='xyxy')

            matched = False
            for track in tracks:
                if iou(adjusted_tlbr.squeeze(0).numpy(), torch.tensor(track.get_tlwh()).unsqueeze(0).numpy()) > 0.5:
                    track.update(det)
                    matched = True
                    break

            if not matched:
                tracks.append(Track(det, score))

        # Prepare the frame for drawing
        frame_draw = convert_image_dtype(frame_tensor.squeeze(0), dtype=torch.uint8)

        # Correctly use adjusted boxes within this loop
        for track in tracks:
            tlwh = track.get_tlwh()
            track_box_adjusted = adjust_boxes([tlwh], scale_factor)[0]
            track_box_tensor = torch.tensor([track_box_adjusted], dtype=torch.float32)
            track_tlbr = box_convert(track_box_tensor, in_fmt='xywh', out_fmt='xyxy')
            frame_draw = draw_bounding_boxes(frame_draw, track_tlbr, colors="blue", width=3)

        # Convert tensor to numpy array for display
        output_image = frame_draw.permute(1, 2, 0).cpu().numpy()
        cv2.imshow('Frame with Tracks', output_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


