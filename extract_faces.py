import cv2
import os
import pandas as pd
import csv
import glob
from download_images import IMAGE_DIR

CROP_DIR = "data/cropped_images"
FAILED_CROPS = "data/failed_crops.csv"
DETECTOR_MODEL = "models/face_detection_yunet_2023mar.onnx"
RECOGNISER_MODEL = "models/face_recognition_sface_2021dec.onnx"
MIN_CERTAINTY = 0.8
MIN_EYE_DISTANCE = 20

def crop(detector_model, recogniser_model, path):
    img = cv2.imread(path)
    if img is None:
        return "unreadable", None
    h, w = img.shape[:2]
    detector_model.setInputSize((w, h))
    _, faces = detector_model.detect(img)
    if faces is None:
        return "no_faces", None
    if len(faces) != 1:
        return "multiple_faces", None
    face = faces[0]
    eye_distance = face[6] - face[4]
    if eye_distance <= MIN_EYE_DISTANCE:
        return "face_too_small", None
    cropped_face = recogniser_model.alignCrop(img, face)
    return "ok", cropped_face

def main():
    detector = cv2.FaceDetectorYN.create(DETECTOR_MODEL, "", (320, 320), MIN_CERTAINTY)
    recogniser = cv2.FaceRecognizerSF.create(RECOGNISER_MODEL, "")
    image_paths = glob.glob(f"{IMAGE_DIR}/*/*.jpg")
    counts = {}
    skipped = 0

    for country in os.listdir(IMAGE_DIR):
        os.makedirs(os.path.join(CROP_DIR, country), exist_ok=True)

    if os.path.exists(FAILED_CROPS):
        failed_people = set(pd.read_csv(FAILED_CROPS)["person"])
    else:
        failed_people = set()

    write_header = not os.path.exists(FAILED_CROPS)
    with open(FAILED_CROPS, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["person", "country", "reason"])

        for path in image_paths:
            parts = path.split("/")
            country = parts[-2]
            person = parts[-1].split(".")[0]
            crop_path = os.path.join(CROP_DIR, country, f"{person}.jpg")
            if os.path.exists(crop_path):
                skipped += 1
                continue
            elif person in failed_people:
                skipped += 1
                continue
            try:
                status, face = crop(detector, recogniser, path)
            except Exception:
                status, face = "error", None
            counts[status] = counts.get(status, 0) + 1
            if status == "ok":
                cv2.imwrite(crop_path, face)
            else:
                writer.writerow([person, country, status])
            if sum(counts.values()) % 1000 == 0:
                print(counts)

    print(f"Done. Skipped {skipped} already processed.")
    print(counts)

if __name__ == "__main__":
    main()
