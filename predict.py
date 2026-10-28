import glob
import time
import cv2
import joblib
import numpy as np
import pandas as pd
from extract_faces import DETECTOR_MODEL, RECOGNISER_MODEL, MIN_CERTAINTY, crop
from train import MODEL_SAVE

SELFIE_DIR = "data/selfies"
TOP_N = 5

MESSAGES = {
    "multiple_faces": "Too many of you in this image! I can only identify one person at a time.",
    "no_faces": "Hmm, I can't seem to find any faces in this image.",
    "face_too_small": "Your face is too small in this photo. Try a closer one!",
    "unreadable": "I can't read this file.",
    "error": "Something went wrong with this image.",
}


def main():
    detector = cv2.FaceDetectorYN.create(DETECTOR_MODEL, "", (320, 320), MIN_CERTAINTY)
    recogniser = cv2.FaceRecognizerSF.create(RECOGNISER_MODEL, "")
    model = joblib.load(MODEL_SAVE)

    counts = pd.read_csv("country_count_1920.csv")
    names = dict(zip(counts["qid"], counts["country"]))

    image_paths = glob.glob(f"{SELFIE_DIR}/*.jpg")
    if len(image_paths) == 0:
        print("No jpegs in the folder.")
        return

    for path in image_paths:
        print(f"\n{path}")
        print("Finding your face...")
        try:
            status, face = crop(detector, recogniser, path)
        except Exception:
            status, face = "error", None

        if status != "ok":
            print(MESSAGES[status])
            continue

        print("So gorgeous. This person is simply glowing today! Preparing my guess...")
        time.sleep(2)

        embedding = recogniser.feature(face)
        X = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
        probabilities = model.predict_proba(X)[0]

        df = pd.DataFrame({"qid": model.classes_, "probability": probabilities})
        df["country"] = df["qid"].map(names)
        df = df.sort_values("probability", ascending=False).head(TOP_N)
        df["probability"] = (df["probability"] * 100).round(1)
        print(df[["country", "probability"]].to_string(index=False))


if __name__ == "__main__":
    main()







    

    

