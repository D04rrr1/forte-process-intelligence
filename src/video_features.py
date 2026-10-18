import cv2
import numpy as np
import torch

from torchvision.models.video import (
    r3d_18,
    R3D_18_Weights,
)


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(
    "Video feature extractor device:",
    DEVICE
)


# ============================================================
# MODEL
# ============================================================

weights = R3D_18_Weights.DEFAULT

model = r3d_18(
    weights=weights
)

# Убираем classifier.
# Получаем 512-dimensional embedding.
model.fc = torch.nn.Identity()

model.eval()
model.to(DEVICE)


# ============================================================
# LOAD VIDEO FRAMES
# ============================================================

def load_video_frames(
    video_path,
    num_frames=16,
    start_sec=None,
    end_sec=None
):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open {video_path}"
        )


    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )


    if fps <= 0:
        cap.release()

        raise RuntimeError(
            f"Invalid FPS: {video_path}"
        )


    # ========================================================
    # TIME RANGE
    # ========================================================

    if start_sec is None:
        start_frame = 0
    else:
        start_frame = int(
            start_sec * fps
        )


    if end_sec is None:
        end_frame = (
            frame_count - 1
        )
    else:
        end_frame = int(
            end_sec * fps
        )


    start_frame = max(
        0,
        min(
            start_frame,
            frame_count - 1
        )
    )


    end_frame = max(
        start_frame,
        min(
            end_frame,
            frame_count - 1
        )
    )


    indices = np.linspace(
        start_frame,
        end_frame,
        num_frames
    ).astype(int)


    frames = []


    for idx in indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(idx)
        )

        ok, frame = cap.read()

        if not ok:
            continue


        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        frame = cv2.resize(
            frame,
            (112, 112)
        )


        frames.append(
            frame
        )


    cap.release()


    if len(frames) == 0:
        raise RuntimeError(
            f"No frames: {video_path}"
        )


    while (
        len(frames)
        <
        num_frames
    ):

        frames.append(
            frames[-1]
        )


    frames = np.array(
        frames
    )


    tensor = torch.tensor(
        frames,
        dtype=torch.float32
    )


    # [T,H,W,C]
    # ->
    # [C,T,H,W]

    tensor = tensor.permute(
        3,
        0,
        1,
        2
    )


    tensor = (
        tensor
        /
        255.0
    )


    # Kinetics normalization

    mean = torch.tensor(
        [
            0.43216,
            0.394666,
            0.37645
        ]
    ).view(
        3,
        1,
        1,
        1
    )


    std = torch.tensor(
        [
            0.22803,
            0.22145,
            0.216989
        ]
    ).view(
        3,
        1,
        1,
        1
    )


    tensor = (
        tensor
        -
        mean
    ) / std


    return tensor


# ============================================================
# EMBEDDING
# ============================================================

def extract_video_embedding(
    video_path,
    start_sec=None,
    end_sec=None,
    num_frames=16
):

    tensor = load_video_frames(
        video_path,
        num_frames=num_frames,
        start_sec=start_sec,
        end_sec=end_sec
    )


    tensor = tensor.unsqueeze(
        0
    )


    tensor = tensor.to(
        DEVICE
    )


    with torch.no_grad():

        embedding = model(
            tensor
        )


    embedding = (
        embedding
        .squeeze(0)
        .cpu()
        .numpy()
    )


    # защита от numerical problems
    embedding = np.nan_to_num(
        embedding,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )


    return embedding