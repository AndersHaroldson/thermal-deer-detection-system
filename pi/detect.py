'''
Real-time thermal deer detection on a Raspberry Pi 4.

Reads frames from a USB thermal camera, runs the NCNN-exported YOLO26n model,
shows the annotated feed full-screen on the LCD, and plays a warning beep once
a deer has been detected in several consecutive frames. Each alert also currently saves
a raw + annotated snapshot so false positives can be reviewed and fed back
into training.
'''

import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR.parent / "models" / "best_ncnn_model"
BEEP_PATH = SCRIPT_DIR / "beep.mp3"
SNAPSHOT_DIR = SCRIPT_DIR / "detections"

# Native resolution of the thermal camera; the model was trained at this size
FRAME_WIDTH = 256
FRAME_HEIGHT = 192

WINDOW_NAME = "Thermal Deer Detection"


def parse_args():
    parser = argparse.ArgumentParser(description="Thermal deer detection for the Raspberry Pi")
    parser.add_argument("--camera", type=int, default=0, help="V4L2 camera index")
    parser.add_argument("--conf", type=float, default=0.75,
                        help="minimum confidence for a frame to count as a detection")
    parser.add_argument("--frames", type=int, default=15,
                        help="consecutive detection frames required before alerting")
    parser.add_argument("--audio-device", default=None,
                        help="ALSA device for mpg123, e.g. hw:1,0 for a USB audio adapter")
    parser.add_argument("--windowed", action="store_true", help="don't run full-screen")
    parser.add_argument("--show-fps", action="store_true", help="draw FPS on the feed")
    return parser.parse_args()


def play_beep(audio_device):
    cmd = ["mpg123", "-q"]
    if audio_device:
        cmd += ["-a", audio_device]
    # Popen so the alert doesn't block detection loop
    subprocess.Popen(cmd + [str(BEEP_PATH)])


def save_snapshot(frame, annotated_frame, timestamp):
    SNAPSHOT_DIR.mkdir(exist_ok=True)
    file_time = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d_%H-%M-%S")
    cv2.imwrite(str(SNAPSHOT_DIR / f"deer_{file_time}_raw.jpg"), frame)
    cv2.imwrite(str(SNAPSHOT_DIR / f"deer_{file_time}_annotated.jpg"), annotated_frame)


def main():
    args = parse_args()

    model = YOLO(str(MODEL_PATH), task="detect")

    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Actual resolution: {actual_width}x{actual_height}")

    if args.windowed:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    else:
        cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Requires several detections in a row so a single noisy frame doesn't trigger an alert
    consec_count = 0
    prev_time = 0
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time else 0
            prev_time = curr_time

            results = model(frame, verbose=False)
            boxes = results[0].boxes
            annotated_frame = results[0].plot()

            if len(boxes) > 0 and boxes.conf.max() > args.conf:
                consec_count += 1
            else:
                consec_count = 0

            if consec_count >= args.frames:
                play_beep(args.audio_device)
                save_snapshot(frame, annotated_frame, curr_time)
                consec_count = 0

            if args.show_fps:
                cv2.putText(annotated_frame, f"FPS: {int(fps)}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow(WINDOW_NAME, annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
