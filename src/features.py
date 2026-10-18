from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights


# Device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print("Feature extractor device:", DEVICE)


# Pretrained ResNet18
weights = ResNet18_Weights.DEFAULT
transform = weights.transforms()

model = resnet18(weights=weights)

# Убираем последний classification layer.
# На выходе останется embedding размером 512.
model = torch.nn.Sequential(
    *list(model.children())[:-1]
)

model.eval()
model.to(DEVICE)


def extract_video_embedding(video_path, num_frames=8):
    """
    Берём num_frames кадров равномерно по всему клипу
    и превращаем их в один 512-dimensional embedding.
    """

    video_path = Path(video_path)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count <= 0:
        cap.release()
        raise RuntimeError(f"No frames: {video_path}")

    indices = np.linspace(
        0,
        frame_count - 1,
        num_frames
    ).astype(int)

    frames = []

    for frame_index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        ok, frame = cap.read()

        if not ok:
            continue

        # OpenCV BGR → RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(frame)
        tensor = transform(image)

        frames.append(tensor)

    cap.release()

    if not frames:
        raise RuntimeError(
            f"Could not extract frames from {video_path}"
        )

    batch = torch.stack(frames).to(DEVICE)

    with torch.no_grad():
        features = model(batch)

    # [frames, 512, 1, 1]
    features = features.squeeze(-1).squeeze(-1)

    # Среднее по кадрам
    embedding = features.mean(dim=0)

    return embedding.cpu().numpy()