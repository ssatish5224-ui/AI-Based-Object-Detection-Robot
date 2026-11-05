import cv2
from ultralytics import YOLO

# AI-Based Object Detection Robot - Software Only
# Uses your laptop webcam and the YOLOv8 nano model.

MODEL_NAME = "yolov8n.pt"
CONFIDENCE = 0.45

def main():
    print("Loading YOLO model... The first run may download yolov8n.pt.")
    model = YOLO(MODEL_NAME)

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("ERROR: Could not open webcam. Check camera permissions or try camera index 1.")
        return

    print("Camera started. Press Q in the camera window to stop.")

    while True:
        success, frame = camera.read()
        if not success:
            print("Could not read a frame from the webcam.")
            break

        results = model.predict(frame, conf=CONFIDENCE, verbose=False)
        annotated_frame = results[0].plot()

        cv2.imshow("AI-Based Object Detection Robot", annotated_frame)

        # Press Q to quit.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
