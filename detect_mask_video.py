import tkinter as tk
from threading import Thread
import cv2
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import imutils

def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)

def start_face_mask_detection():
    def detect():
        while not stop_detection:
            frame = vs.read()
            frame = imutils.resize(frame, width=400)
            (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

            for (box, pred) in zip(locs, preds):
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                stop_detection_thread()

    Thread(target=detect, daemon=True).start()

def stop_detection_thread():
    global stop_detection
    stop_detection = True

# Load face detector and mask detector models
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
maskNet = load_model("mask_detector.model")

# Initialize the video stream
vs = VideoStream(src=0).start()

# Create the main window
root = tk.Tk()
root.title("Face Mask Detection")

# Create a button to start face mask detection
start_button = tk.Button(root, text="Detect Face Mask", command=start_face_mask_detection)
start_button.pack(pady=10)

# Create a button to stop face mask detection
stop_button = tk.Button(root, text="Stop Detection", command=stop_detection_thread)
stop_button.pack(pady=10)

# Create a label to display instructions or information
info_label = tk.Label(root, text="Press 'Detect Face Mask' to start face mask detection.")
info_label.pack(pady=10)

# Flag to control the face mask detection thread
stop_detection = False

# Run the Tkinter main loop
root.mainloop()
 
# Cleanup code (outside the Tkinter main loop)
cv2.destroyAllWindows()
vs.stop()
