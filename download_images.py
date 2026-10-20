import os
import time
import pandas as pd
import requests
from count_countries import HEADERS

PEOPLE_FILE = "data/people.csv"
IMAGE_DIR = "data/images"
FAILED_FILE = "data/failed_downloads.csv"
IMG_WIDTH = 500
DELAY = 0.5
IMAGE_HEADERS = {"User-Agent": HEADERS["User-Agent"]}


def download(image_url, path):
    response = requests.get(
        image_url,
        params={"width": IMG_WIDTH},
        headers=IMAGE_HEADERS,
        timeout=30,
    )
    response.raise_for_status()
    temp_path = path + ".part"
    with open(temp_path, "wb") as f:
        f.write(response.content)
    os.replace(temp_path, path)


def main():
    people = pd.read_csv(PEOPLE_FILE)
    print(f"{len(people)} people to download\n")

    for qid in people["qid"].unique():
        os.makedirs(os.path.join(IMAGE_DIR, qid), exist_ok=True)

    downloaded = 0
    skipped = 0
    failed = []

    for i, (person, qid, image_url) in enumerate(
        zip(people["person"], people["qid"], people["image_url"]), start=1
    ):
        path = os.path.join(IMAGE_DIR, qid, f"{person}.jpg")
        if os.path.exists(path):
            skipped += 1
            continue

        success = False
        for attempt in range(1, 4):
            try:
                download(image_url, path)
                success = True
                break
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code
                if status == 429:
                    print("  Too many requests: pausing for 60 seconds")
                    time.sleep(60)
                elif status == 404:
                    break
                else:
                    time.sleep(5 * attempt)
            except Exception:
                time.sleep(5 * attempt)

        if success:
            downloaded += 1
        else:
            failed.append(person)

        if i % 100 == 0:
            print(f"{i}/{len(people)}  downloaded {downloaded}, skipped {skipped}, failed {len(failed)}")

        time.sleep(DELAY)

    print(f"\nDone. Downloaded {downloaded}, skipped {skipped}, failed {len(failed)}")
    pd.Series(failed, name="person").to_csv(FAILED_FILE, index=False)


if __name__ == "__main__":
    main()

