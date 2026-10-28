# trace-my-face
 
Can a classifier tell which country someone is from by looking at their face?
 
Basically no. Across 73 countries it gets **16.1% balanced accuracy against a 1.4% chance baseline**. This is about 11× better than guessing, but it's also wrong five times out of six. Given ethnic similarities and immigration, it's pretty difficult or perhaps impossible for detectors like this to be very accurate. Perhaps a neural network would be slightly better but not by much.
 
## Pipeline
 
Labels come from Wikidata (SPARQL) rather than image search, meaning every label is attached to a specific person rather than search engine fluff. Trained and tested on 172,531 people → ~172k Wikimedia thumbnails → **132,946** aligned 112×112 face crops (YuNet for detection, SFace for alignment and 128-dim embeddings) → logistic regression on L2-normalised embeddings, 80/20 stratified, class-weighted.
 
| | |
|---|---|
| `count_countries.py` | people per country available on Wikidata |
| `fetch_people.py` | people + image links → `data/people.csv` |
| `download_images.py` | thumbnails (~2 days, rate-limited, resumable) |
| `extract_faces.py` | detect, align, crop |
| `embed_faces.py` | → `data/embeddings.npz` |
| `train.py` / `predict.py` | train and evaluate / top-5 on a new photo |
 
Key decisions: born 1920+ (otherwise European counts are filled with painted portraits, statues etc. The filter halves France but cuts India by only 24%); birth years capped at 2004 so everyone is definitely an adult; ~350 people per country per decade; dual citizens dropped entirely; one image per person, so leakage between train and test is impossible. 77% of images yielded a usable face, whilst the rest were rejected for multiple faces (11%), too-small faces (8%) or none detected (4%). `predict.py` imports `crop` directly from `extract_faces.py` to stop preprocessing drifting between training and inference.
 
## Results
 
| Metric | Score | Chance |
|---|---|---|
| Top-1 | 14.7% | 1.4% |
| **Balanced** | **16.1%** | 1.4% |
| Top-5 | 39.2% | 6.8% |
 
The spread across countries is enormous: Ghana 59.9%, Bangladesh 54.8%, South Korea 52.8%, Nigeria 45.5%, Indonesia 45.1% — against Germany **0.9%** (below chance, on 464 test faces), Slovenia 1.4%, Belgium 2.0%, Austria 2.1%, UK 2.9%.
 
Rather than this being about certain populations having more distinct features, it's probably a function of how many near neighbours a country has among the other 72 classes. A German face almost always has an Austrian, Swiss, Dutch or Czech class fitting equally well, meaning thaat the model rarely commits to Germany at all. On the other hand, Ghana has few lookalikes in this particular set. High-immigrant societies like the US (7.7%), Canada (4.3%), Brazil (5.3%) have citizens of every ancestry, so there's no single appearance to learn. Nationality is a legal fact, and a model can only really ever approximate broad ancestry clusters correlating loosely with it. The top-5 jump from 16% to 39% seems to fit this theory: the model is often almost but not quite right, landing somewhere in the correct cluster.
 
## Running it
 
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install requests pandas opencv-python scikit-learn
mkdir -p models && wget -P models \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface_2021dec.onnx
export WIKIDATA_CONTACT="your@email.com"   # Wikimedia requires real contact details
```
 
Run the scripts in order. Wikimedia only serves [standard thumbnail widths](https://www.mediawiki.org/wiki/Common_thumbnail_sizes) without aggressive rate limiting. `data/` is gitignored; the scripts regenerate everything.
 
## No website?

The model is super inaccurate but I still wouldn't want to do this for ethical reasons. This is intended more as a fun tool to play around with rather than something that should be taken seriously!
 
Code MIT. Wikidata is CC0; Commons images carry individual licences; YuNet and SFace are from OpenCV Zoo under their own terms.
