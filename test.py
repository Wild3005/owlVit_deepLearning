import cv2
import torch
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection

processor = OwlViTProcessor.from_pretrained(
    "google/owlvit-base-patch32"
)

model = OwlViTForObjectDetection.from_pretrained(
    "google/owlvit-base-patch32"
)

texts = [[
    "human face",
    "sunglasses",
    "face mask",
    "hat"
]]

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image = Image.fromarray(rgb)

    inputs = processor(
        text=texts,
        images=image,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])

    results = processor.post_process_object_detection(
        outputs=outputs,
        threshold=0.2,
        target_sizes=target_sizes
    )[0]

    for box, score, label in zip(
        results["boxes"],
        results["scores"],
        results["labels"]
    ):

        x1, y1, x2, y2 = box.tolist()

        name = texts[0][label]

        cv2.rectangle(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"{name} {score:.2f}",
            (int(x1), int(y1)-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,255,0),
            2
        )

    cv2.imshow("OWL-ViT", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()