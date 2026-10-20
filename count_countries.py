import time
import requests 
import pandas as pd
import os

ENDPOINT = "https://query.wikidata.org/sparql"

HEADERS = {
    "User-Agent": f"face-origin-learning-project/0.1 ({os.environ['WIKIDATA_CONTACT']}",
    "Accept": "application/sparql-results+json",
}

EXTRA_IDS = {
    "Q29999": ["Q55"], # Kingdom of the Netherlands (also Netherlands)
    "Q756617": ["Q35"], # Kingdom of Denmark (also Denmark)
    "Q145": ["Q21", "Q22", "Q25", "Q26"] # United Kingdom (also its four constituent countries)
}

def run_query(query): 
    response = requests.get(
        ENDPOINT,
        params={"query": query},
        headers=HEADERS,
        timeout=90,
    )
    response.raise_for_status()
    return response.json()["results"]["bindings"]

# Query 1: all current sovereign states (English names) 

COUNTRIES_QUERY = """
SELECT ?country ?countryLabel WHERE {
?country wdt:P31 wd:Q3624078 .
FILTER NOT EXISTS { ?country wdt:P576 ?dissolved . }
SERVICE wikibase:label {bd:serviceParam wikibase:language "en". }
}
"""

# Query 2: how many humans with a photo are citizens of one country 

COUNT_QUERY = """
SELECT (COUNT(DISTINCT ?person) AS ?n) WHERE {
VALUES ?c { COUNTRY_IDS }
?person wdt:P27 ?c ;
wdt:P31 wd:Q5 ;
wdt:P18 ?image ;
wdt:P569 ?born .
FILTER(YEAR(?born) >= 1920)
}
"""

def main():
    countries = run_query(COUNTRIES_QUERY)
    print(f"Found {len(countries)} countries")

    rows = []
    for i, c in enumerate(countries, start=1):
        qid = c["country"]["value"].split("/")[-1]
        name = c["countryLabel"]["value"]
        all_ids = [qid] + EXTRA_IDS.get(qid, [])
        values = " ".join(f"wd:{x}" for x in all_ids)

        n = None
        for attempt in range(1, 4):
            try:
                result = run_query(COUNT_QUERY.replace("COUNTRY_IDS", values))
                n = int(result[0]["n"]["value"])
                break
            except Exception as e:
                print(f" {name}: attempt {attempt} failed ({e})")
                time.sleep(5 * attempt)


        print(f"[{i}/{len(countries)}] {name}: {n}")

        rows.append({"qid": qid, "country": name, "people_with_photo": n})

        time.sleep(1)

    df = pd.DataFrame(rows).sort_values("people_with_photo", ascending=False)
    df.to_csv("country_count_1920.csv", index=False)

    print("\nAll countries")
    print(df.to_string(index=False))

    print(f"\nCountries with 200+ people: {(df['people_with_photo'] >= 200).sum()}")

if __name__ == "__main__":
    main()


