import cv2
import os
import pandas as pd
import csv
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
    return "successful crop", cropped_face

def main():

if __name__ == "__main__":
    main()
