import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import random
import tempfile

model = YOLO("currencymodel.pt")
st.title("Jordanian Currency detection model", anchor=False)
classes = list(model.names.values())


def annotate_image(image, results):
    for box in results[0].boxes:  # Extract bounding boxes from the results
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
        conf = box.conf[0]  # Confidence score of the detection
        cls_id = int(box.cls[0])  # Class ID of the detected object
        # if you want to show conf level add to the label {conf:.2f}
        label = f"{results[0].names[cls_id]}"  # Label with class name and confidence
        color = class_colors[results[0].names[cls_id]]  # Color for the class

        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 12)

        # Calculate the text size and position
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 4, 2)
        cv2.rectangle(
            image, (x1, y1 - h - 10), (x1 + w, y1), color, -1
        )  # Background rectangle for label
        cv2.putText(
            image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 2
        )  # Text label in black color

    return image


# Function to generate distinct colors for each class
def generate_colors(num_colors):
    return [tuple(random.randint(0, 255) for _ in range(3)) for _ in range(num_colors)]


# Generate a unique color for each class and map them
class_colors = {
    cls: color for cls, color in zip(classes, generate_colors(len(classes)))
}


# Image upload and processing
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_image:
    image = Image.open(uploaded_image)  # Open the uploaded image

    # Convert PIL image to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Run inference on the image
    results = model(image_cv)

    # Annotate the image with detections
    annotated_image = annotate_image(image_cv, results)

    # Convert annotated image back to RGB for displaying in Streamlit
    st.image(
        cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB),
        caption="Detected Objects.",
        use_column_width=True,
    )

    # Convert annotated image to PIL format
    annotated_image_pil = Image.fromarray(
        cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    )

    # Create a download button for the annotated image
    st.download_button(
        label="Download Annotated Image",
        data=cv2.imencode(
            ".jpg", cv2.cvtColor(np.array(annotated_image_pil), cv2.COLOR_RGB2BGR)
        )[1].tobytes(),
        file_name="annotated_image.jpg",
        mime="image/jpeg",
    )


# # Video upload and processing
# uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
# if uploaded_video:
#     # Save uploaded video to a temporary file
#     tfile = tempfile.NamedTemporaryFile(delete=False)
#     tfile.write(uploaded_video.read())

#     # Open the video file
#     video_cap = cv2.VideoCapture(tfile.name)
#     stframe = st.empty()  # Placeholder for displaying video frames

#     # Initialize VideoWriter for saving the output video
#     output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
#     out = cv2.VideoWriter(
#         output_video_path,
#         cv2.VideoWriter_fourcc(*"mp4v"),
#         20.0,
#         (
#             int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
#             int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
#         ),
#     )

#     while video_cap.isOpened():
#         ret, frame = video_cap.read()  # Read a frame from the video
#         if not ret:
#             break

#         # Run inference on the frame
#         results = model(frame)
#         annotated_frames = []
#         annotated_frames.append(frame)
#         # Annotate the frame with detections
#         annotated_frame = annotate_image(frame, results)

#         # Write the annotated frame to the output video
#         out.write(annotated_frame)

#         # Convert annotated frame back to RGB for displaying in Streamlit(if you want the video to be shown before the download button appears)
#         # stframe.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_column_width=True)

#     video_cap.release()  # Release the video capture object
#     out.release()  # Release the VideoWriter object
#     # Create a download button for the annotated video
#     with open(output_video_path, "rb") as video_file:
#         st.download_button(
#             label="Download Annotated Video",
#             data=video_file,
#             file_name="annotated_video.mp4",
#             mime="video/mp4",
#         )


# import cv2
# import streamlit as st
# from ultralytics import YOLO
# import json
# import tempfile
# import os
# import numpy as np
# from PIL import Image
# import random

# # Load model
# model = YOLO("C:/Users/HP/OneDrive/Desktop/IMP_Projects/CV/currencymodel.pt")


# # Function to perform inference and draw bounding boxes
# def detect_currency(input_image):

#     image = Image.open(input_image)  # Open the uploaded image

#     # Convert PIL image to OpenCV format
#     image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

#     # Run inference
#     results = model(input_image)

#     # Convert results to JSON
#     results_json = json.loads(results[0].tojson())

#     # Draw bounding boxes on the image
#     for detection in results_json:
#         box = detection["box"]
#         x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
#         confidence = detection["confidence"]
#         label = f"{detection['name']} ({confidence:.2f})"

#         # Draw the box
#         cv2.rectangle(input_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         # Put the label
#         cv2.putText(
#             input_image,
#             label,
#             (x1, y1 - 10),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.9,
#             (0, 255, 0),
#             2,
#         )

#     # Convert annotated image back to RGB for displaying in Streamlit
#     st.image(
#         cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB),
#         caption="Detected Objects.",
#         use_column_width=True,
#     )

#     # Convert annotated image to PIL format
#     annotated_image_pil = Image.fromarray(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))

#     return annotated_image_pil


# # Streamlit app
# def main():
#     st.title("Currency Detection App")

#     uploaded_file = st.file_uploader(
#         "Choose an image or video", type=["jpg", "png", "jpeg", "mp4"]
#     )

#     output_image = detect_currency(uploaded_file)

#     # Create a download button for the annotated image
#     st.download_button(
#         label="Download Annotated Image",
#         data=cv2.imencode(
#             ".jpg", cv2.cvtColor(np.array(output_image), cv2.COLOR_RGB2BGR)
#         )[1].tobytes(),
#         file_name="annotated_image.jpg",
#         mime="image/jpeg",
#     )


# if __name__ == "__main__":
#     main()


# # cd C:\Users\HP\OneDrive\Desktop\work\AI portfolio\AI
# # streamlit run app2.py
