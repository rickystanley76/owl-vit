import cv2
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import numpy as np

# Load the processor and model
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

# Define the text queries
texts = [["player", "ball","person", "child"]]

# Define a dictionary to map labels to colors
label_colors = {
    0: "red",  # cat
    1: "green",
    2: "yellow",
    3: "cyan"# dog

}

# Open the video file
video_path = r"C:/Users/A549773/ML-examples-RD/TRUSTY/OWL-ViT/source/baba-ethan.mp4"
cap = cv2.VideoCapture(video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to a PIL image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

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

    # Convert the PIL image back to a NumPy array
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Write the frame to the output video
    out.write(frame)

    # Display the frame (optional)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()