import cv2
import torch

from PIL import Image

from torchvision.models.detection import (
    ssdlite320_mobilenet_v3_large,
    SSDLite320_MobileNet_V3_Large_Weights
)


# Пока detection запускаем на CPU.
# Для первого этапа этого достаточно и стабильнее.
DEVICE = torch.device("cpu")

print("Detector device:", DEVICE)


weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT

model = ssdlite320_mobilenet_v3_large(
    weights=weights
)

model.eval()
model.to(DEVICE)

transform = weights.transforms()


def detect_people(frame, threshold=0.5):
    """
    Получает OpenCV frame (BGR).

    Возвращает список:
    [
        {
            "box": [x1, y1, x2, y2],
            "score": 0.95
        }
    ]
    """

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    image = Image.fromarray(rgb)

    tensor = transform(image)
    tensor = tensor.to(DEVICE)

    with torch.no_grad():
        prediction = model(
            [tensor]
        )[0]

    people = []

    boxes = prediction["boxes"].cpu().numpy()
    labels = prediction["labels"].cpu().numpy()
    scores = prediction["scores"].cpu().numpy()

    for box, label, score in zip(
        boxes,
        labels,
        scores
    ):

        # COCO class 1 = person
        if label != 1:
            continue

        if score < threshold:
            continue

        x1, y1, x2, y2 = box.astype(int)

        people.append({
            "box": [
                int(x1),
                int(y1),
                int(x2),
                int(y2)
            ],
            "score": float(score)
        })

    return people