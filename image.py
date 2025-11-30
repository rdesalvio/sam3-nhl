from PIL import Image
import matplotlib.pyplot as plt
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Load SAM3 model
model = build_sam3_image_model(device="cuda")
processor = Sam3Processor(model)

image = Image.open("pictures/bedard.jpg").convert('RGB')
inference_state = processor.set_image(image)

text_prompt = "player in white jersey"  # You can input any concept
output = processor.set_text_prompt(state=inference_state, prompt=text_prompt)

# Get segmentation results
masks = output["masks"]        # Segmentation masks
boxes = output["boxes"]        # Bounding boxes
scores = output["scores"]      # Confidence scores

# Convert tensors to numpy arrays (move from GPU to CPU if needed)
if hasattr(boxes, 'cpu'):
    boxes = boxes.cpu().numpy()
if hasattr(scores, 'cpu'):
    scores = scores.cpu().numpy()

# Display results - simple version showing bounding boxes on original image
def show_results(image, boxes, scores, text_prompt, output_path="players_labelled.png"):
    plt.figure(figsize=(12, 8))
    plt.imshow(image)

    # Draw each bounding box
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                           fill=False, color='red', linewidth=3)
        plt.gca().add_patch(rect)

        # Add score label
        plt.text(x1, y1-10, f'{score:.2f}',
                bbox=dict(facecolor='red', alpha=0.7),
                fontsize=12, color='yellow')

    plt.title(f'Text Prompt: "{text_prompt}"', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"Results saved to {output_path}")
    plt.close()

show_results(image, boxes, scores, text_prompt)