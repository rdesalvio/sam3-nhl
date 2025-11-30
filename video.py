import torch
import cv2
import numpy as np
from sam3.model_builder import build_sam3_video_predictor

# Initialize SAM3 video predictor
gpus_to_use = [torch.cuda.current_device()]
video_predictor = build_sam3_video_predictor(gpus_to_use)

# Set video path (can be JPEG folder or MP4 file)
video_path = "videos/short_example.mp4"
output_video_path = "output_video.mp4"

# Start video segmentation session
response = video_predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path=video_path,
    )
)

session_id = response["session_id"]
print(f"Video session started, ID: {session_id}")

# Add text prompt at frame 0
frame_index = 0
text_prompt = "player in white jersey"

response = video_predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=frame_index,
        text=text_prompt,
    )
)

print(f"Found {len(response['outputs'])} matching objects in frame {frame_index}")

# Propagate masks across all video frames
print("Propagating masks across all frames...")
outputs_per_frame = {}
for response in video_predictor.handle_stream_request(
    request=dict(
        type="propagate_in_video",
        session_id=session_id,
    )
):
    outputs_per_frame[response["frame_index"]] = response["outputs"]
    if response["frame_index"] % 30 == 0:
        print(f"Processed frame {response['frame_index']}")

print(f"Propagation complete! Processed {len(outputs_per_frame)} frames")

# Helper function to convert mask to bounding box
def mask_to_bbox(mask):
    """Convert binary mask to bounding box [x1, y1, x2, y2]"""
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    # Find non-zero pixels
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        return None

    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]

    return [int(x1), int(y1), int(x2), int(y2)]

# Open input video to get properties
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

print(f"Creating output video: {output_video_path}")

# Process each frame
frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Get outputs for this frame (if available)
    if frame_idx in outputs_per_frame:
        outputs = outputs_per_frame[frame_idx]

        # Handle the actual SAM3 output structure
        # outputs is a dict with keys: out_obj_ids, out_probs, out_boxes_xywh, out_binary_masks
        if isinstance(outputs, dict) and 'out_obj_ids' in outputs:
            obj_ids = outputs.get('out_obj_ids', [])
            probs = outputs.get('out_probs', [])
            boxes_xywh = outputs.get('out_boxes_xywh', [])
            masks = outputs.get('out_binary_masks', [])

            # Debug on first frame
            if frame_idx == 0:
                print(f"\nFrame 0 debug info:")
                print(f"  obj_ids type: {type(obj_ids)}, shape: {obj_ids.shape if hasattr(obj_ids, 'shape') else 'N/A'}")
                print(f"  probs type: {type(probs)}, shape: {probs.shape if hasattr(probs, 'shape') else 'N/A'}")
                print(f"  boxes_xywh type: {type(boxes_xywh)}, shape: {boxes_xywh.shape if hasattr(boxes_xywh, 'shape') else 'N/A'}")
                print(f"  obj_ids: {obj_ids}")
                print(f"  boxes_xywh: {boxes_xywh}")

            # Convert to numpy if needed
            if hasattr(obj_ids, 'cpu'):
                obj_ids = obj_ids.cpu().numpy()
            if hasattr(probs, 'cpu'):
                probs = probs.cpu().numpy()
            if hasattr(boxes_xywh, 'cpu'):
                boxes_xywh = boxes_xywh.cpu().numpy()

            # Debug after conversion
            if frame_idx == 0:
                print(f"\nAfter numpy conversion:")
                print(f"  obj_ids length: {len(obj_ids)}")
                print(f"  boxes_xywh length: {len(boxes_xywh)}")

            # Draw bounding boxes for each detected object
            num_drawn = 0
            for i in range(len(obj_ids)):
                object_id = obj_ids[i]
                score = probs[i] if i < len(probs) else 1.0

                # Get bounding box in xywh format and convert to xyxy
                if i < len(boxes_xywh):
                    x, y, w, h = boxes_xywh[i]

                    # Coordinates are normalized (0-1), scale to image dimensions
                    x1 = int(x * width)
                    y1 = int(y * height)
                    x2 = int((x + w) * width)
                    y2 = int((y + h) * height)

                    # Draw bounding box
                    color = (0, 255, 0)  # Green
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Draw label with object ID and score
                    label = f"ID:{object_id} {score:.2f}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    thickness = 2

                    # Get text size for background
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, font, font_scale, thickness
                    )

                    # Draw background rectangle for text
                    cv2.rectangle(frame,
                                (x1, y1 - text_height - 10),
                                (x1 + text_width, y1),
                                color, -1)

                    # Draw text
                    cv2.putText(frame, label, (x1, y1 - 5),
                              font, font_scale, (0, 0, 0), thickness)

                    num_drawn += 1

            # Debug: print how many boxes were drawn on first frame
            if frame_idx == 0:
                print(f"  Drew {num_drawn} bounding boxes on frame 0")

    # Write frame to output video
    out.write(frame)
    frame_idx += 1

    if frame_idx % 30 == 0:
        print(f"Writing frame {frame_idx}/{len(outputs_per_frame)}")

# Release resources
cap.release()
out.release()

print(f"Done! Output video saved to: {output_video_path}")
print(f"Text prompt: '{text_prompt}'")
print(f"Total frames processed: {frame_idx}")
