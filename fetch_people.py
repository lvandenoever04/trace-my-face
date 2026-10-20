import time 
import pandas as pd 
import os
from count_countries import EXTRA_IDS, run_query

COUNTS_FILE = "country_count_1920.csv"
OUTPUT_FILE = "data/people.csv"
MIN_PEOPLE = 1000
PER_DECADE_LIMIT = 350
DECADES = range(1920, 2010, 10)
MAX_BIRTH_YEAR = 2004
CAP_THRESHOLD = 3000

PEOPLE_QUERY = """
SELECT ?person (SAMPLE(?image) AS ?img) (MIN(YEAR(?born)) AS ?year) WHERE {
VALUES ?c { COUNTRY_IDS }
?person wdt:P27 ?c ;
wdt:P31 wd:Q5 ;
wdt:P18 ?image ;
wdt:P569 ?born .
FILTER(YEAR(?born) >= START_YEAR && YEAR(?born) <= END_YEAR)
}
GROUP BY ?person
LIMIT MAX_PEOPLE
"""

def query_with_retries(query, attempts=3):
    for attempt in range(1, attempts + 1):
        try:
            return run_query(query)
        except Exception as e:
            print(f" attempt {attempt} failed ({e})")
            if attempt < attempts:
                time.sleep(5 * attempt)
    return None

def main():
    counts = pd.read_csv(COUNTS_FILE)
    countries = counts[counts["people_with_photo"] >= MIN_PEOPLE]
    print(f"Fetching people for {len(countries)} countries\n")

    people = []
    failed = []

    for i, (qid, name, total) in enumerate(
        zip(countries["qid"], countries["country"], countries["people_with_photo"]), start=1,
        ):
        all_ids = [qid] + EXTRA_IDS.get(qid, [])
        values = " ".join(f"wd:{x}" for x in all_ids)
        limit = PER_DECADE_LIMIT if total > CAP_THRESHOLD else int(total)
        print(f"[{i}/{len(countries)}] {name}")
        for start in DECADES:
            end = min(start + 9, MAX_BIRTH_YEAR)

            query = (
            PEOPLE_QUERY
            .replace("COUNTRY_IDS", values)
            .replace("START_YEAR", str(start))
            .replace("END_YEAR", str(end))
            .replace("MAX_PEOPLE", str(limit))
            )
            results = query_with_retries(query)

            if results is None:
                print(f" {start}s failed")
                failed.append((name, start))
                continue

            for row in results:
                people.append({
                    "person": row["person"]["value"].split("/")[-1],
                    "qid": qid,
                    "country": name,
                    "birth_year": int(row["year"]["value"]),
                    "image_url": row["img"]["value"],
                })

            print(f"  {start}s: {len(results)} people")
            time.sleep(1)
    df = pd.DataFrame(people)
    print(f"\n Raw rows: {len(df)}")

    df = df.drop_duplicates(subset=["person", "qid"])
    print(f"After removing duplicates: {len(df)}")

    df = df[~df.duplicated(subset="person", keep=False)]
    print(f"After removing dual citizens: {len(df)}")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(df)} people to {OUTPUT_FILE}")

    print("\nPeople per country:")
    print(df["country"].value_counts().to_string())

    if failed:
        print(f"\n{len(failed)} queries failed:")
        for country, decade in failed:
            print(f"  {country}, {decade}s")
    else:
        print("\nNo failed queries.")


if __name__ == "__main__":
    main()