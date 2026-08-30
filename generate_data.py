"""
Generate the synthetic incident dataset this application runs on.

    python generate_data.py            # writes data/focus_df.csv
    python generate_data.py --seed 7   # a different synthetic city
    python generate_data.py --records 5000

WHAT IS REAL AND WHAT IS NOT
----------------------------
Real, and kept because it is public record:

  * Barangay names and boundaries (angeles_city_barangays.geojson) come from
    the standard Philippine administrative boundary dataset.
  * Barangay population, land area and density (data/angeles_city_other_info.csv)
    come from Philippine Statistics Authority census figures.
  * Police station coordinates are public locations.

Synthetic, generated here:

  * Every crime incident. Dates, times, offenses, coordinates, case outcomes,
    and all suspect and victim demographics are fabricated by this script.

The real incident data came from the Angeles City Police Office under a
confidentiality agreement and is not distributable. It appears nowhere in this
repository. Figures produced from this generated data are therefore not the
figures reported in the paper -- see README.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from geopy.distance import geodesic

GEOJSON = Path("angeles_city_barangays.geojson")
DEMOGRAPHICS = Path("data") / "angeles_city_other_info.csv"
OUT = Path("data") / "focus_df.csv"

# Public locations of the six Angeles City police stations.
POLICE_STATIONS = {
    "Police Station 1": (15.1350, 120.5910),
    "Police Station 2": (15.1614, 120.6087),
    "Police Station 3": (15.1607, 120.6089),
    "Police Station 4": (15.1600, 120.5918),
    "Police Station 5": (15.1446, 120.5572),
    "Police Station 6": (15.15884, 120.59226),
}

# The eight focus crimes the model classifies, with the free-text offense
# descriptions the pipeline's regex layer normalises back into them. Weights are
# invented but long-tailed, so property crime dominates the way it usually does.
FOCUS_CRIMES = {
    "Theft":             (0.34, ["THEFT", "QUALIFIED THEFT"], "INDEX CRIME"),
    "Robbery":           (0.19, ["ROBBERY", "ROBBERY WITH VIOLENCE"], "INDEX CRIME"),
    "Rape":              (0.11, ["RAPE", "ANTI-RAPE LAW OF 1997",
                                 "ACT PROMOTING STRONGER PROTECTION AGAINST RAPE"], "NON INDEX CRIME"),
    "Carnapping MC":     (0.10, ["CARNAPPING MC"], "INDEX CRIME"),
    "Homicide":          (0.08, ["HOMICIDE", "FRUSTRATED HOMICIDE"], "INDEX CRIME"),
    "Murder":            (0.07, ["MURDER", "PARRICIDE"], "INDEX CRIME"),
    "Physical Injuries": (0.06, ["SERIOUS PHYSICAL INJURIES",
                                 "SLIGHT PHYSICAL INJURIES"], "INDEX CRIME"),
    "Carnapping MV":     (0.05, ["CARNAPPING MV"], "INDEX CRIME"),
}

AGE_BANDS = [("0_17", 0, 17), ("18_25", 18, 25), ("26_34", 26, 34), ("35_44", 35, 44),
             ("45_54", 45, 54), ("55_64", 55, 64), ("65_Above", 65, 200)]

HOUR_LABELS = ["Midnight", "Morning", "Afternoon", "Evening"]


def barangay_centroids(path: Path) -> dict[str, tuple[float, float]]:
    """Average each barangay polygon's vertices so incidents land inside it."""
    geo = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for feat in geo["features"]:
        pts = []
        def walk(c):
            if isinstance(c[0], (int, float)):
                pts.append(c)
            else:
                for sub in c:
                    walk(sub)
        walk(feat["geometry"]["coordinates"])
        arr = np.array(pts, dtype=float)
        out[feat["properties"]["ADM4_EN"]] = (arr[:, 1].mean(), arr[:, 0].mean())
    return out


def hour_weights() -> np.ndarray:
    """Bimodal day: a morning bump, a heavier evening peak, a quiet small hours."""
    h = np.arange(24)
    w = 0.20 + 0.75 * np.exp(-0.5 * ((h - 9) / 2.6) ** 2) \
             + 1.35 * np.exp(-0.5 * ((h - 20) / 3.1) ** 2)
    return w / w.sum()


def band_counts(ages: list[int], gender_flags: list[str]) -> dict:
    """Age-band and gender breakdown for one incident's people."""
    out = {f"{label}": sum(1 for a in ages if lo <= a <= hi) for label, lo, hi in AGE_BANDS}
    out["male"] = sum(1 for g in gender_flags if g == "Male")
    out["female"] = sum(1 for g in gender_flags if g == "Female")
    return out


def generate(seed: int, n_records: int, start: str, end: str) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    demo = pd.read_csv(DEMOGRAPHICS)
    centroids = barangay_centroids(GEOJSON)

    # The demographics CSV uses the crime-dataset spelling; the geojson uses
    # title case. Match them loosely so centroids resolve for every barangay.
    def find_centroid(name: str):
        key = name.upper().replace(".", "").replace("(POB)", "").strip()
        for gname, coord in centroids.items():
            g = gname.upper().replace(".", "")
            if key in g or g.split("(")[0].strip() in key:
                return coord
        return (15.1450, 120.5900)

    names = demo["Barangay"].tolist()
    coords = {n: find_centroid(n) for n in names}

    # Incident volume tracks population, with noise so the model cannot simply
    # read the count off the population column.
    w = demo["Population_2024"].to_numpy(float) * rng.lognormal(0, 0.45, len(demo))
    w /= w.sum()
    idx = rng.choice(len(demo), size=n_records, p=w)

    days = (pd.Timestamp(end) - pd.Timestamp(start)).days
    day_w = np.linspace(1.25, 0.75, days + 1)      # mild downward trend
    day_w /= day_w.sum()
    dates = pd.Timestamp(start) + pd.to_timedelta(
        rng.choice(days + 1, size=n_records, p=day_w), unit="D")

    hours = rng.choice(24, size=n_records, p=hour_weights())
    minutes = rng.choice([0, 15, 30, 45], size=n_records, p=[0.55, 0.12, 0.23, 0.10])

    labels = list(FOCUS_CRIMES)
    probs = np.array([FOCUS_CRIMES[k][0] for k in labels]); probs /= probs.sum()
    focus = rng.choice(labels, size=n_records, p=probs)

    rows = []
    for i in range(n_records):
        bgy = names[idx[i]]
        d = dates[i]
        f = focus[i]
        offense = rng.choice(FOCUS_CRIMES[f][1])
        clat, clon = coords[bgy]
        lat = round(clat + rng.normal(0, 0.004), 6)
        lon = round(clon + rng.normal(0, 0.004), 6)

        station = rng.choice(list(POLICE_STATIONS))
        dists = {s: geodesic((lat, lon), c).km for s, c in POLICE_STATIONS.items()}
        nearest = min(dists, key=dists.get)

        n_sus = int(rng.choice([0, 1, 2, 3, 4], p=[.06, .68, .16, .07, .03]))
        n_vic = int(rng.choice([0, 1, 2, 3], p=[.04, .84, .09, .03]))
        sus_ages = [int(np.clip(rng.normal(31, 12), 14, 78)) for _ in range(n_sus)]
        vic_ages = [int(np.clip(rng.normal(33, 15), 3, 88)) for _ in range(n_vic)]
        sus_g = list(rng.choice(["Male", "Female"], n_sus, p=[.78, .22]))
        vic_g = list(rng.choice(["Male", "Female"], n_vic, p=[.44, .56]))
        sb, vb = band_counts(sus_ages, sus_g), band_counts(vic_ages, vic_g)

        hour = int(hours[i])
        demo_row = demo.iloc[idx[i]]

        row = {
            "Offense ID": i + 1,
            "Barangay": bgy,
            "Date": d.strftime("%d/%m/%Y"),
            "Time Committed": f"{hour}:{int(minutes[i]):02d}:00",
            "Offense Committed": offense,
            "Focus_Crime": f,
            "Crime Type": FOCUS_CRIMES[f][2],
            "Case Status": rng.choice(["Solved", "Cleared", "Under Investigation"],
                                      p=[.84, .10, .06]),
            "Latitude": lat, "Longitude": lon,
            "Victim Count": n_vic, "Suspect Count": n_sus,
            "Year": d.year, "Month": d.month, "Day": d.day,
            "Day_of_Week": d.day_name(), "Hour": hour,
            "Weekday": d.isoweekday(),
            "Week_of_Year": int(d.isocalendar().week), "Quarter": d.quarter,
            "Is_Weekend": int(d.isoweekday() >= 6),
            "Time_of_Day": HOUR_LABELS[min(hour // 6, 3)],
            "Police Station": station,
            "Distance_from_Police": dists[station],
            "Nearest_Police_Station": nearest,
            "Nearest_Police_Distance": dists[nearest],
            "Num_Police_Stations_1km": sum(1 for v in dists.values() if v <= 1.0),
            "Num_Suspects": float(n_sus),
            "Avg_Suspects_Age": float(np.mean(sus_ages)) if sus_ages else 0.0,
            "Male_Suspects": float(sb["male"]), "Female_Suspects": float(sb["female"]),
            "Num_Victims": float(n_vic),
            "Avg_Victims_Age": float(np.mean(vic_ages)) if vic_ages else 0.0,
            "Male_Victims": float(vb["male"]), "Female_Victims": float(vb["female"]),
        }
        for label, _, _ in AGE_BANDS:
            row[f"Suspects_{label}"] = float(sb[label])
            row[f"Victims_{label}"] = float(vb[label])
        for col in ["Area_sqm", "Area_sqkm", "Population_2020", "Population_2024",
                    "Pop_Density_2020", "Pop_Density_2024", "Pop_Growth_Rate"]:
            row[col] = demo_row[col]
        rows.append(row)

    # Column order matches what app.py and xgboost_model.py expect.
    order = ["Offense ID", "Barangay", "Date", "Time Committed", "Offense Committed",
             "Focus_Crime", "Crime Type", "Case Status", "Latitude", "Longitude",
             "Victim Count", "Suspect Count", "Year", "Month", "Day", "Day_of_Week",
             "Hour", "Weekday", "Week_of_Year", "Quarter", "Is_Weekend", "Time_of_Day",
             "Police Station", "Distance_from_Police", "Nearest_Police_Station",
             "Nearest_Police_Distance", "Num_Police_Stations_1km",
             "Num_Suspects", "Avg_Suspects_Age", "Male_Suspects", "Female_Suspects"] + \
            [f"Suspects_{l}" for l, _, _ in AGE_BANDS] + \
            ["Num_Victims", "Avg_Victims_Age", "Male_Victims", "Female_Victims"] + \
            [f"Victims_{l}" for l, _, _ in AGE_BANDS] + \
            ["Area_sqm", "Area_sqkm", "Population_2020", "Population_2024",
             "Pop_Density_2020", "Pop_Density_2024", "Pop_Growth_Rate"]
    return pd.DataFrame(rows)[order]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--records", type=int, default=8000)
    ap.add_argument("--start", default="2017-01-01")
    ap.add_argument("--end", default="2024-12-31")
    args = ap.parse_args()

    print(f"Generating {args.records:,} synthetic incidents (seed {args.seed})...")
    df = generate(args.seed, args.records, args.start, args.end)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print(f"Wrote {OUT}  {df.shape}")
    print(f"  barangays   {df['Barangay'].nunique()}")
    print(f"  date range  {args.start} to {args.end}")
    print(f"  crime mix   {df['Focus_Crime'].value_counts().to_dict()}")
    print("\nNext: python xgboost_model.py    (trains on this data)")
    print("      python app.py")


if __name__ == "__main__":
    main()
