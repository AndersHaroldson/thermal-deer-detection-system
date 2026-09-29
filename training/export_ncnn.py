'''
Export trained weights to NCNN for fast CPU inference on the Raspberry Pi.

Usage:
    python export_ncnn.py --weights runs/detect/deer/weights/best.pt

Writes a best_ncnn_model/ folder next to the weights; copy it to models/ on the Pi.
'''

import argparse

from ultralytics import YOLO

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export the deer detector to NCNN")
    parser.add_argument("--weights", required=True, help="path to trained .pt weights")
    args = parser.parse_args()

    model = YOLO(args.weights)
    model.export(format="ncnn", imgsz=[192, 256])
