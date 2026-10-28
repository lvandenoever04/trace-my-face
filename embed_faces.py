import cv2
import glob
import numpy as np
from extract_faces import CROP_DIR, RECOGNISER_MODEL

EMBEDDINGS_FILE = "data/embeddings.npz"

def main():
    recogniser = cv2.FaceRecognizerSF.create(RECOGNISER_MODEL, "")
    image_paths = glob.glob(f"{CROP_DIR}/*/*.jpg")
    print(f"{len(image_paths)} faces found\n")

    embeddings = []
    countries = []
    people = []

    for path in image_paths:
        parts = path.split("/")
        country = parts[-2]
        person = parts[-1].split(".")[0]
        img = cv2.imread(path)
        embedding = recogniser.feature(img)

        embeddings.append(embedding[0])
        countries.append(country)
        people.append(person)

        if len(embeddings) % 10000 == 0:
            print(f"{len(embeddings)} faces embedded.")

    np.savez(
        EMBEDDINGS_FILE,
        embeddings=np.array(embeddings),
        countries=np.array(countries),
        people=np.array(people),
    )
    print(f"Saved {len(embeddings)} embeddings to {EMBEDDINGS_FILE}")

if __name__ == "__main__":
    main()