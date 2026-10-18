import torch

from torchvision.models.detection import (
    keypointrcnn_resnet50_fpn,
    KeypointRCNN_ResNet50_FPN_Weights
)


# Для detection-моделей CPU сейчас надёжнее,
# особенно на Mac.
DEVICE = torch.device("cpu")


weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT

model = keypointrcnn_resnet50_fpn(
    weights=weights
)

model.to(DEVICE)
model.eval()


# COCO keypoints:
# 0 nose
# 5 left shoulder
# 6 right shoulder
# 7 left elbow
# 8 right elbow
# 9 left wrist
# 10 right wrist

LEFT_WRIST = 9
RIGHT_WRIST = 10


def detect_poses(
    frame,
    score_threshold=0.7,
    keypoint_threshold=0.5
):
    """
    Возвращает список людей с bbox
    и координатами запястий.
    """

    image = (
        torch.from_numpy(frame)
        .permute(2, 0, 1)
        .float()
        / 255.0
    )

    # OpenCV BGR -> RGB
    image = image[[2, 1, 0]]

    image = image.to(DEVICE)

    with torch.no_grad():
        output = model([image])[0]

    results = []

    for i in range(
        len(output["scores"])
    ):

        score = float(
            output["scores"][i]
        )

        if score < score_threshold:
            continue

        box = (
            output["boxes"][i]
            .cpu()
            .numpy()
            .astype(int)
            .tolist()
        )

        keypoints = (
            output["keypoints"][i]
            .cpu()
            .numpy()
        )

        left = keypoints[
            LEFT_WRIST
        ]

        right = keypoints[
            RIGHT_WRIST
        ]

        left_wrist = None
        right_wrist = None

        if left[2] >= keypoint_threshold:
            left_wrist = (
                int(left[0]),
                int(left[1])
            )

        if right[2] >= keypoint_threshold:
            right_wrist = (
                int(right[0]),
                int(right[1])
            )

        results.append({
            "box": box,
            "score": score,
            "left_wrist": left_wrist,
            "right_wrist": right_wrist
        })

    return results