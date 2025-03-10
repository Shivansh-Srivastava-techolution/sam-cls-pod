import cv2
import os
import time
import threading
from datetime import datetime
from collections import deque
import tkinter as tk
from tkinter import simpledialog

def decode_fourcc(value):
    return "".join([chr((value >> 8 * i) & 0xFF) for i in range(4)])

def configure_camera(cap, width=1280, height=720, fps=30, codec="MJPG"):
    if not cap or not cap.isOpened():
        return None

    fourcc = cv2.VideoWriter_fourcc(*codec)
    old_fourcc = decode_fourcc(int(cap.get(cv2.CAP_PROP_FOURCC)))

    if cap.set(cv2.CAP_PROP_FOURCC, fourcc):
        print(f"Codec changed from {old_fourcc} to {decode_fourcc(int(cap.get(cv2.CAP_PROP_FOURCC)))}")
    else:
        print(f"Error: Could not change codec from {old_fourcc}.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    print(f"Camera configured with FPS: {cap.get(cv2.CAP_PROP_FPS)}, "
          f"Width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}, "
          f"Height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

    return cap

def capture_frame(frame, output_folder="data"):
    if frame is not None:
        root = tk.Tk()
        root.withdraw()  # Hide main window
        label = simpledialog.askstring("Input", "Enter the ref number for the captured object:")
        if label:
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            filename = os.path.join(output_folder, f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Frame captured: {filename}")
        return filename, label

def main():
    camera_id = 1
    width = 3840
    height = 2160
    fps = 30

    cap = cv2.VideoCapture(camera_id)
    cap = configure_camera(cap, width, height, fps)
    
    if not cap:
        print("Error: Could not open camera.")
        return

    fps_values = deque(maxlen=30)
    last_time = time.perf_counter()
    
    cv2.namedWindow("Video Capture", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Video Capture", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame. Exiting.")
            break

        frame = cv2.rotate(frame, cv2.ROTATE_180)
        
        current_time = time.perf_counter()
        frame_time = current_time - last_time
        fps = 1 / frame_time if frame_time > 0 else 0
        fps_values.append(fps)
        avg_fps = sum(fps_values) / len(fps_values)
        last_time = current_time

        cv2.imshow("Video Capture", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            filename, label = capture_frame(frame)
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
