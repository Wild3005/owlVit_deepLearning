"""
ATM Security System – OWL-ViT Based Face Coverage Detector
============================================================
Logika keamanan:
  - Deteksi: human face, sunglasses, face mask, hat
  - Wajah HARUS terdeteksi  → jika tidak ada wajah, AKSES DITOLAK
  - Jika wajah terdeteksi tapi ADA kacamata hitam / masker / topi
    yang overlap dengan area wajah  → AKSES DITOLAK (wajah tertutup)
  - Jika wajah terdeteksi dan TIDAK ada atribut penutup → AKSES DIIZINKAN

Cara pakai:
  python atm_security.py --image images/input2.png
  python atm_security.py --image images/input.jpg
  python atm_security.py --camera          # gunakan webcam
"""

import argparse
import sys
import os
from datetime import datetime

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection


# ─────────────────────────────────────────────────────────────────────────────
# Konfigurasi
# ─────────────────────────────────────────────────────────────────────────────

DETECTION_THRESHOLD   = 0.13   # confidence minimum OWL-ViT
IOU_OVERLAP_THRESHOLD = 0.10   # minimal overlap antara face & atribut penutup
FACE_COVER_RATIO      = 0.40   # jika bbox cover < threshold ini → wajah dianggap tertutup

# ── Query Set 1: label utama (urutan menjadi indeks label) ────────────────────
LABELS_PRIMARY = [
    "human face",   # 0
    "sunglasses",   # 1
    "face mask",    # 2
    "hat",          # 3
]

# ── Query Set 2: alias lebih deskriptif untuk masker & topi ──────────────────
LABELS_ALT = [
    "surgical mask covering mouth and nose",   # → mapped ke face mask (2)
    "medical face mask",                        # → face mask (2)
    "cloth face mask",                          # → face mask (2)
    "baseball cap worn on head",                # → hat (3)
    "cap covering forehead",                    # → hat (3)
    "beanie hat",                               # → hat (3)
    "dark sunglasses covering eyes",            # → sunglasses (1)
]

# Mapping dari indeks ALT → indeks PRIMARY
ALT_TO_PRIMARY = {
    0: 2,   # surgical mask     → face mask
    1: 2,   # medical face mask → face mask
    2: 2,   # cloth face mask   → face mask
    3: 3,   # baseball cap      → hat
    4: 3,   # cap covering      → hat
    5: 3,   # beanie hat        → hat
    6: 1,   # dark sunglasses   → sunglasses
}

# Label index yang dianggap "penutup wajah"
COVER_LABEL_IDS = {1, 2, 3}   # sunglasses, face mask, hat

# Warna bounding box per kategori (BGR)
COLORS = {
    "human face":  (0, 220,  0),    # hijau
    "sunglasses":  (0,  60, 255),   # merah-oranye
    "face mask":   (0,  60, 255),
    "hat":         (0,  60, 255),
}

# Label nama gabungan (index 0-3 saja yang dipakai di luar)
LABELS = LABELS_PRIMARY


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_iou(boxA, boxB):
    """
    Hitung Intersection over Union antara dua bounding box.
    Format box: [x1, y1, x2, y2]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    if inter_area == 0:
        return 0.0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union_area = areaA + areaB - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def nms(detections, iou_threshold=0.50):
    """
    Non-Maximum Suppression — hapus deteksi duplikat yang overlap.
    Pertahankan deteksi dengan confidence tertinggi.
    """
    if not detections:
        return []
    detections = sorted(detections, key=lambda d: d["score"], reverse=True)
    kept = []
    for det in detections:
        dominated = False
        for k in kept:
            if k["label_id"] == det["label_id"] and compute_iou(k["box"], det["box"]) > iou_threshold:
                dominated = True
                break
        if not dominated:
            kept.append(det)
    return kept


def _infer_once(image_pil, label_list, threshold):
    """Satu pass inference OWL-ViT dengan satu set label."""
    texts = [label_list]
    inputs = processor(text=texts, images=image_pil, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([image_pil.size[::-1]])
    try:
        results = processor.post_process_grounded_object_detection(
            outputs=outputs, threshold=threshold, target_sizes=target_sizes
        )[0]
    except AttributeError:
        results = processor.post_process_object_detection(
            outputs=outputs, threshold=threshold, target_sizes=target_sizes
        )[0]
    return results


def run_detection(image_pil):
    """
    Multi-query OWL-ViT inference:
      Pass 1 → LABELS_PRIMARY  (human face, sunglasses, face mask, hat)
      Pass 2 → LABELS_ALT      (deskripsi alternatif masker & topi)
    Hasil digabung → NMS → return list deteksi terkonsolidasi.
    """
    detections = []

    # ── Pass 1: label primer ──────────────────────────────────────────────
    r1 = _infer_once(image_pil, LABELS_PRIMARY, DETECTION_THRESHOLD)
    for box, score, label in zip(r1["boxes"], r1["scores"], r1["labels"]):
        lid = label.item()
        detections.append({
            "label_id":   lid,
            "label_name": LABELS_PRIMARY[lid],
            "score":      score.item(),
            "box":        [int(v) for v in box.tolist()],
        })

    # ── Pass 2: label alternatif (masker & topi variasi) ──────────────────
    r2 = _infer_once(image_pil, LABELS_ALT, DETECTION_THRESHOLD)
    for box, score, label in zip(r2["boxes"], r2["scores"], r2["labels"]):
        alt_id    = label.item()
        prim_id   = ALT_TO_PRIMARY[alt_id]
        detections.append({
            "label_id":   prim_id,
            "label_name": LABELS_PRIMARY[prim_id],
            "score":      score.item(),
            "box":        [int(v) for v in box.tolist()],
        })

    # ── NMS per-class untuk deduplicate ───────────────────────────────────
    detections = nms(detections, iou_threshold=0.40)

    # ── Pass 3: OpenCV fallback ───────────────────────────────────────
    # Jika OWL-ViT belum mendeteksi masker/topi, gunakan CV klasik sebagai fallback
    faces = [d for d in detections if d["label_id"] == 0]
    already_has_covers = any(d["label_id"] in COVER_LABEL_IDS for d in detections)
    if faces and not already_has_covers:
        frame_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        cv_covers = cv_fallback_covers(frame_bgr, faces)
        detections.extend(cv_covers)
        if cv_covers:
            detections = nms(detections, iou_threshold=0.40)

    return detections


def cv_detect_mask(frame_bgr, face_box):
    """
    Deteksi masker wajah menggunakan analisis warna HSV.
    Cek region sepertiga bawah bounding box wajah:
      - Masker medis putih: S rendah, V tinggi
      - Masker kain gelap / berwarna: S tinggi
    Return True jika kemungkinan ada masker.
    """
    x1, y1, x2, y2 = face_box
    # Region of interest: sepertiga bawah wajah (area mulut-hidung)
    roi_y1 = y1 + (y2 - y1) * 2 // 3
    roi_y2 = y2
    roi    = frame_bgr[roi_y1:roi_y2, x1:x2]
    if roi.size == 0:
        return False
    hsv   = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Masker putih/terang: V > 150, S < 60
    white_mask = cv2.inRange(hsv, (0,  0, 150), (180,  60, 255))
    # Masker biru medis: H 90-130, S > 50
    blue_mask  = cv2.inRange(hsv, (90, 50,  50), (130, 255, 255))
    # Masker hitam: V < 60
    black_mask = cv2.inRange(hsv, (0,   0,   0), (180,  255,  60))
    combined   = cv2.bitwise_or(white_mask, cv2.bitwise_or(blue_mask, black_mask))
    total_px   = roi.shape[0] * roi.shape[1]
    cover_ratio = cv2.countNonZero(combined) / total_px if total_px > 0 else 0
    return cover_ratio > 0.45   # >45% piksel region bawah wajah = masker


def cv_detect_hat(frame_bgr, face_box, img_h):
    """
    Deteksi topi berdasarkan posisi wajah:
    Jika bagian atas wajah (y1) dekat dengan batas atas gambar
    DAN ada region gelap / berwarna di atas batas wajah → kemungkinan topi.
    Return True jika kemungkinan ada topi.
    """
    x1, y1, x2, y2 = face_box
    face_h = y2 - y1

    # Jika tidak ada ruang di atas wajah untuk dicek
    if y1 <= 5:
        return False

    # Region atas wajah (area di atas dahi)
    brim_y1 = max(0, y1 - face_h // 2)
    brim_y2 = y1
    brim    = frame_bgr[brim_y1:brim_y2, x1:x2]
    if brim.size == 0:
        return False

    # Cek apakah ada objek berwarna solid di atas wajah (brim topi)
    gray   = cv2.cvtColor(brim, cv2.COLOR_BGR2GRAY)
    _, thr = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)   # area gelap
    dark_ratio = cv2.countNonZero(thr) / (brim.shape[0] * brim.shape[1])

    # Brim topi biasanya mengisi >35% area di atas wajah
    # Juga: jika y1 sangat dekat dengan batas atas gambar, wajah tertutup topi
    head_top_ratio = y1 / img_h   # seberapa tinggi wajah dari atas gambar
    return dark_ratio > 0.35 or head_top_ratio < 0.10


def cv_fallback_covers(frame_bgr, faces):
    """
    Fallback deteksi atribut penutup berbasis OpenCV klasik.
    Dipakai jika OWL-ViT tidak mendeteksi penutup apapun.
    Return list deteksi sintetik dengan label_id & box.
    """
    img_h, img_w = frame_bgr.shape[:2]
    results = []
    for face in faces:
        # Cek masker
        if cv_detect_mask(frame_bgr, face["box"]):
            x1, y1, x2, y2 = face["box"]
            roi_y1 = y1 + (y2 - y1) * 2 // 3
            results.append({
                "label_id":   2,
                "label_name": "face mask",
                "score":      0.60,   # confidence sintetik
                "box":        [x1, roi_y1, x2, y2],
                "source":     "cv_fallback",
            })
        # Cek topi
        if cv_detect_hat(frame_bgr, face["box"], img_h):
            x1, y1, x2, y2 = face["box"]
            brim_y1 = max(0, y1 - (y2 - y1) // 2)
            results.append({
                "label_id":   3,
                "label_name": "hat",
                "score":      0.55,
                "box":        [x1, brim_y1, x2, y1],
                "source":     "cv_fallback",
            })
    return results


def face_visibility_ratio(face_box, img_w, img_h):
    """
    Hitung rasio tinggi bounding-box wajah terhadap tinggi gambar.
    Jika topi menekan kepala ke bawah, hanya bagian bawah wajah yang
    muncul → box vertikal sempit → rasio rendah.
    Threshold: jika rasio < FACE_COVER_RATIO → wajah dianggap tersembunyi.
    """
    x1, y1, x2, y2 = face_box
    face_h = y2 - y1
    face_w = x2 - x1
    # Rasio area relatif terhadap ukuran gambar
    face_area  = face_h * face_w
    image_area = img_h  * img_w
    return face_area / image_area if image_area > 0 else 1.0


def atm_security_logic(detections, img_w=None, img_h=None):
    """
    Tentukan apakah transaksi ATM diizinkan.

    Return:
        status  : "ALLOWED" | "DENIED"
        reason  : string alasan keputusan
        faces   : list deteksi wajah
        covers  : list deteksi atribut penutup
    """
    faces  = [d for d in detections if d["label_id"] == 0]
    covers = [d for d in detections if d["label_id"] in COVER_LABEL_IDS]

    # ── Aturan 1: Wajah harus terdeteksi ─────────────────────────────────
    if not faces:
        return (
            "DENIED",
            "Wajah tidak terdeteksi – identitas tidak dapat diverifikasi",
            faces,
            covers,
        )

    # ── Aturan 2: Cek tumpang-tindih atribut penutup dengan wajah ────────
    for face in faces:
        for cover in covers:
            iou = compute_iou(face["box"], cover["box"])
            if iou >= IOU_OVERLAP_THRESHOLD:
                reason = (
                    f"Wajah tertutup oleh '{cover['label_name']}' "
                    f"(IoU={iou:.2f}, conf={cover['score']:.2f}). "
                    "Lepas atribut penutup wajah sebelum bertransaksi."
                )
                return "DENIED", reason, faces, covers

    # ── Aturan 3: Cek wajah terlalu kecil (topi menekan kepala, wajah tersembunyi)
    # Berlaku bahkan jika OWL-ViT tidak mendeteksi penutup, karena
    # cv_fallback sudah meng-handle itu; ini safety net tambahan.
    if img_w and img_h:
        for face in faces:
            ratio = face_visibility_ratio(face["box"], img_w, img_h)
            if ratio < FACE_COVER_RATIO:
                # Dapatkan nama penutup jika ada
                cover_names = ", ".join(set(c["label_name"] for c in covers)) if covers else "atribut tidak teridentifikasi"
                reason = (
                    f"Wajah hanya terlihat sebagian (visibility={ratio:.2f}) "
                    f"kemungkinan tertutup oleh {cover_names}. "
                    "Pastikan seluruh wajah terlihat dengan jelas."
                )
                return "DENIED", reason, faces, covers

    # ── Aturan 4: Wajah terlihat jelas, tidak ada penutup ────────────────
    return (
        "ALLOWED",
        "Wajah terdeteksi dengan jelas. Transaksi dapat dilanjutkan.",
        faces,
        covers,
    )



def draw_results(frame, detections, status, reason):
    """Gambar bounding box dan banner status ATM di atas frame."""
    h, w = frame.shape[:2]

    # ── Bounding box ──────────────────────────────────────────────────────
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        color = COLORS.get(d["label_name"], (200, 200, 200))
        lbl   = f"{d['label_name']} {d['score']:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, lbl, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # ── Banner status ATM ─────────────────────────────────────────────────
    banner_h     = 90
    banner_color = (0, 130, 0)   if status == "ALLOWED" else (30, 30, 180)
    status_text  = "TRANSAKSI DIIZINKAN" if status == "ALLOWED" else "TRANSAKSI DITOLAK"
    icon         = "v" if status == "ALLOWED" else "X"

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), banner_color, -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    cv2.putText(frame, f"[{icon}] {status_text}",
                (16, 44), cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 2)

    # Tulis reason (dibagi ke beberapa baris jika panjang)
    max_chars = max(40, w // 13)
    words, line, reason_lines = reason.split(), "", []
    for word in words:
        if len(line) + len(word) + 1 <= max_chars:
            line = (line + " " + word).strip()
        else:
            reason_lines.append(line)
            line = word
    if line:
        reason_lines.append(line)

    for i, rl in enumerate(reason_lines[:2]):
        cv2.putText(frame, rl, (16, 66 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (210, 210, 210), 1)

    # Timestamp
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, f"ATM Security | {ts}",
                (w - 330, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Mode gambar statis
# ─────────────────────────────────────────────────────────────────────────────

def process_image(image_path: str, no_display: bool = False):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Gagal membaca: {image_path}")
        sys.exit(1)

    rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(rgb)

    print(f"[INFO] Mendeteksi objek pada: {image_path}")
    detections = run_detection(image_pil)

    img_h, img_w = frame.shape[:2]

    print(f"[INFO] Jumlah deteksi: {len(detections)}")
    for d in detections:
        print(f"       [{d['label_name']:35s}]  conf={d['score']:.3f}  box={d['box']}")

    status, reason, faces, covers = atm_security_logic(detections, img_w, img_h)

    print(f"\n{'='*62}")
    print(f"  STATUS  : {status}")
    print(f"  Alasan  : {reason}")
    print(f"{'='*62}\n")

    frame = draw_results(frame, detections, status, reason)

    os.makedirs("save_images", exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"save_images/atm_result_{ts}.jpg"
    cv2.imwrite(out_path, frame)
    print(f"[INFO] Hasil disimpan: {out_path}")

    if not no_display:
        cv2.imshow("ATM Security System – OWL-ViT", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
# Mode kamera realtime
# ─────────────────────────────────────────────────────────────────────────────

def process_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Kamera tidak ditemukan.")
        sys.exit(1)

    print("[INFO] Mode kamera aktif. Tekan 'q' atau ESC untuk keluar.")

    last_detections = []
    last_status     = "DENIED"
    last_reason     = "Menginisialisasi sistem…"
    frame_count     = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Jalankan inference setiap 5 frame agar lebih responsif
        if frame_count % 5 == 0:
            rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(rgb)
            last_detections            = run_detection(image_pil)
            h, w = frame.shape[:2]
            last_status, last_reason, _, _ = atm_security_logic(last_detections, w, h)

        frame = draw_results(frame, last_detections, last_status, last_reason)
        cv2.imshow("ATM Security System – OWL-ViT (Live)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):   # 'q' atau ESC
            break

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ATM Security System – deteksi wajah & atribut penutup (OWL-ViT)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  "-i", type=str,
                       help="Path file gambar (contoh: images/input2.png)")
    group.add_argument("--camera", "-c", action="store_true",
                       help="Gunakan webcam secara realtime")
    parser.add_argument("--no-display", "-n", action="store_true",
                        help="Simpan hasil ke file tanpa membuka jendela GUI")
    args = parser.parse_args()

    # Muat model OWL-ViT (sekali saja)
    print("[INFO] Memuat model OWL-ViT… (pertama kali: unduh ~1 GB)")
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model     = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
    model.eval()
    print("[INFO] Model siap.\n")

    if args.camera:
        process_camera()
    else:
        process_image(args.image, no_display=args.no_display)
