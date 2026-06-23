import cv2
import torch
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from datetime import datetime

# Load model
processor = OwlViTProcessor.from_pretrained(
    "google/owlvit-base-patch32"
)

model = OwlViTForObjectDetection.from_pretrained(
    "google/owlvit-base-patch32"
)

# Prompt yang ingin dicari
texts = [[
    "human face",
    "sunglasses",
    "face mask",
    "hat"
]]

# Load gambar
image_path = "images/input.jpg"

frame = cv2.imread(image_path)

if frame is None:
    raise FileNotFoundError(f"Gagal membaca gambar: {image_path}")

rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

image = Image.fromarray(rgb)

# Inference
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

# Gambar bounding box
for box, score, label in zip(
    results["boxes"],
    results["scores"],
    results["labels"]
):

    x1, y1, x2, y2 = box.tolist()

    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)

    name = texts[0][label]

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"{name} {score:.2f}",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

# Simpan hasil
# output_path = "save_images/hasil_deteksi.jpg"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = f"save_images/hasil_deteksi_{timestamp}.jpg"

cv2.imwrite(output_path, frame)

print(f"Hasil disimpan: {output_path}")

# Tampilkan hasil
cv2.imshow("OWL-ViT Result", frame)

cv2.waitKey(0)
cv2.destroyAllWindows()