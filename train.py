import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from embed_faces import EMBEDDINGS_FILE

TEST_SIZE = 0.2
RANDOM_SEED = 21

def main():
    data = np.load(EMBEDDINGS_FILE)
    X = data["embeddings"]
    y = data["countries"]
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED,
    )
    print(f"Training on {len(X_train)} faces, testing on {len(X_test)}")

    print("Training...")
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X_train, y_train)

    print("Testing...")
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    balanced_accuracy = balanced_accuracy_score(y_test, predictions)

    rows = []
    countries = np.unique(y)
    for country in countries:
        mask = y_test == country
        country_predictions = predictions[mask]
        correct = country_predictions == country
        rows.append({"qid": country, "accuracy": correct.mean(), "test_faces": mask.sum()})

    counts = pd.read_csv("country_count_1920.csv")
    names = dict(zip(counts["qid"], counts["country"]))
    df = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    df["country"] = df["qid"].map(names)
    df["accuracy"] = (df["accuracy"] * 100).round(1)
    print(df[["country", "accuracy", "test_faces"]].to_string(index=False))

    print(f"Accuracy: {accuracy:.1%}. Balanced accuracy: {balanced_accuracy:.1%}")
    print(f"Chance level: {1/len(np.unique(y)):.1%}")

if __name__ == "__main__":
    main()

