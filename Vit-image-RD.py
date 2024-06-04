from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import matplotlib.pyplot as plt

# Load the processor and model
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

# Provide the path to your local image
image_path = r"C:/Users/A549773/ML-examples-RD/TRUSTY/OWL-ViT/source/field.jpg"
image = Image.open(image_path)

# Define the text queries
texts = [["child", "cap", "bottle", "cup", "plate", "food", "tree", "house", "stone"]]

# Process the image and text
inputs = processor(text=texts, images=image, return_tensors="pt")
outputs = model(**inputs)

# Get target sizes and process object detection results
target_sizes = torch.Tensor([image.size[::-1]])
results = processor.post_process_object_detection(
    outputs=outputs, target_sizes=target_sizes, threshold=0.1
)

# Retrieve predictions for the first image
i = 0
text = texts[i]
boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

# Create a draw object
draw = ImageDraw.Draw(image)
font = ImageFont.truetype("arial.ttf", 50)  # Use a larger font size

# Define a dictionary to map labels to colors https://matplotlib.org/stable/gallery/color/named_colors.html 
label_colors = {
    0: "red",  # cat
    1: "green",  # dog
    2: "blue",  # remote control
    3: "yellow",
    4: "darkorange",
    5: "brown",
    6: "cyan",
    7: "indigo",
    8: "lime"
}

for box, score, label in zip(boxes, scores, labels):
    box = [round(i, 2) for i in box.tolist()]
    label_text = f"{text[label.item()]}: {round(score.item(), 2)}"
    print(f"Detected {label_text} at location {box}")
    
    # Convert the label tensor to an integer
    label_int = label.item()
    
    # Draw the bounding box with a specific color
    color = label_colors.get(label_int, "black")  # Use "black" as a default color if label_int is not in label_colors
    draw.rectangle(box, outline=color, width=2)
    
    # Calculate text size
    text_bbox = draw.textbbox((0, 0), label_text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    # Draw a filled rectangle behind the text
    draw.rectangle(
        [box[0], box[1] - text_height, box[0] + text_width, box[1]],
        fill=color
    )
    draw.text((box[0], box[1] - text_height), label_text, fill="white", font=font)

# Display the image using matplotlib
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis("off")
plt.show()