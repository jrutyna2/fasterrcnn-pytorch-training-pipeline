# 22 minutes to  run CPU
#
import os
import torch
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from custom_dataset import CustomDataset
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the pre-trained model
model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
model = model.to(device)
model.eval()

# Define the transform function
def transform_function(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform(image)

# Prepare the data loader
# dataset = CustomDataset('/content/drive/MyDrive/Dataset_fasterrcnn/test', transform=transform_function)
dataset = CustomDataset('data/', transform=transform_function)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

# COCO class labels
COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Run inference and visualize results
for i, (images, image_paths) in enumerate(data_loader):
    print(f"Processing image {i+1}/{len(dataset)}")
    images = list(img.to(device) for img in images)
    with torch.no_grad():
        predictions = model(images)

    # Process predictions for each image
    for j, (prediction, image_path) in enumerate(zip(predictions, image_paths)):
        # Extract original image name
        original_image_name = os.path.splitext(os.path.basename(image_path))[0]

        # Draw bounding boxes and labels
#        image = Image.fromarray(images[j].mul(255).permute(1, 2, 0).byte().numpy())
        image = Image.fromarray(images[j].mul(255).permute(1, 2, 0).byte().cpu().numpy())

        draw = ImageDraw.Draw(image)

        if not os.path.exists('results_untrained'):
            os.makedirs('results_untrained')

        # Directory for saving results
        results_dir = 'results_untrained'

        # Check if the directory exists, and if not, create it
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # Open a text file to save bounding box data
        with open(f'results_untrained/{original_image_name}.txt', 'w') as f:
            for element in range(len(prediction['boxes'])):
                boxes = prediction['boxes'][element].cpu().numpy()
                score = prediction['scores'][element].cpu().numpy()
                label_id = prediction['labels'][element].cpu().numpy()
                if 0 < label_id <= len(COCO_LABELS):
                    label = COCO_LABELS[label_id - 1]
                else:
                    label = "Unknown"
                if score > 0.5:  # Threshold
                    # Write bounding box data to file
                    f.write(f"Label: {label}, Score: {score:.2f}, Box: {boxes.tolist()}\n")

                    # Draw bounding box and label on image
                    draw.rectangle([(boxes[0], boxes[1]), (boxes[2], boxes[3])], outline="red")
                    label_text = f"{label}: {score:.2f}"
                    draw.text((boxes[0], boxes[1]), text=label_text)

        # Save the image with original name
        image.save(f'results_untrained/{original_image_name}.png')
