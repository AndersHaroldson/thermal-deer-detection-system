'''
Fine-tune YOLO26n on the thermal deer dataset.

Trained on a desktop GPU at the thermal camera's native 256x192 resolution

Usage:
    python train.py --data path/to/deer-set/data.yaml
'''

import argparse

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train the thermal deer detector")
    parser.add_argument("--data", required=True, help="path to the dataset's data.yaml")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="0", help="CUDA device index, or 'cpu'")
    parser.add_argument("--name", default="deer", help="run name under runs/detect/")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    model = YOLO('yolo26n.pt')
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=[192, 256],
        rect=True,
        single_cls=False,
        mosaic=0.5,
        mixup=0.0,
        batch=args.batch,
        device=args.device,
        name=args.name,
    )
