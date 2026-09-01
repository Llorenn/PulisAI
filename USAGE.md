# Using PulisAI

How to run the application and work each of its three pages. For what the
project is and how the model works, see [README.md](README.md).

---

## Starting it

```bash
pip install -r requirements.txt
python generate_data.py      # creates data/focus_df.csv
python xgboost_model.py      # trains the model on it
python app.py                # http://127.0.0.1:5000
```

The first two steps are needed once. `data/focus_df.csv` and the `.joblib`
model are not in the repository — they are generated, so a fresh clone has to
build them before `app.py` will start.

If you skip `generate_data.py`, the app still starts but every page is empty and
the console prints `Error: The dataset file 'data/focus_df.csv' was not found.`
If you skip `xgboost_model.py`, the pages render but predictions return
`Model is not loaded`.

### Signing in

Default is `admin` / `pulisai`. Override before running:

```bash
export PULISAI_USER=yourname
export PULISAI_PASSWORD=yourpassword
export PULISAI_SECRET_KEY=$(python -c "import secrets;print(secrets.token_hex(32))")
```

Set `PULISAI_SECRET_KEY` if you want sessions to survive a restart — without it
a new key is generated each time the app starts, which logs everyone out.

Every page except the login screen requires a session. Visiting any URL while
signed out redirects to `/login`.

---

## Crime Hotspot — predicting alarm levels

The landing page after login. Produces a Low / Medium / High risk level for
barangays under a specific set of conditions.

**To make a prediction:**

1. **Barangay** — checkbox dropdown, choose one or many. "Select All" toggles
   every barangay at once.
2. **Month**, **Weekday**, **Time of Day** — one value each.
3. Click **PREDICT**.

All four are required; the form blocks submission and lists what is missing.

**Reading the map.** Only the barangays you selected are coloured. Everything
else stays light grey — that is "not selected", not "no risk". Colours are
green Low, amber Medium, red High.

Click any coloured barangay for its population, land area, crime density per
km², most frequent offense and peak hour. Hovering thickens the outline.

**One thing to watch.** A barangay with no historical record for the exact
combination you picked is shown as **Low, in green**, with `N/A` in the popup
fields. The four selectors span 11,088 possible combinations and the lookup
table holds only a few thousand, so most specific queries fall through to this
default. If the popup reads `N/A` for peak hour and most frequent crime, the
green means "nothing on file", not "assessed as safe".

**Forecast.** After predicting, a twelve-month projection appears below the map,
covering whichever barangays you selected. It needs more than 12 months of
history or it reports insufficient data.

**CLEAR** resets the form and the map.

---

## Dashboard — exploring the incident set

Filter the data across six dimensions and read the result in eight metric tiles
and six linked charts.

**Filters:** crime type, year, month, weekday, time of day (all multi-select
checkbox dropdowns), plus a start/end date range. Leaving a dropdown empty means
no filtering on it. Click **APPLY FILTERS**; **CLEAR** resets everything.

**Tiles:** top barangay, top crime, total crimes, solved, under investigation,
cleared, total suspects, total victims.

**Charts:** an incident scatter map, a table of the first 50 matching records, a
victim age-group pie, a year-over-year trend line, and bar charts by time of day
and by day of week. All are Plotly — drag to zoom, double-click to reset, hover
for values, and use the toolbar to download a PNG.

Filters are carried in the URL, so a filtered view can be bookmarked or shared.

---

## Database — uploading data and retraining

Extends the model with new records without touching code.

The page opens on a browsable table of the current dataset with its own search
and column filters. Below it are three file inputs.

**Upload requires all three CSVs together:**

| Field | Must contain |
|---|---|
| Crime data | `Offense ID`, `Barangay`, `Date`, `Time Committed`, `Offense Committed`, `Crime Type`, `Case Status`, `Latitude`, `Longitude`, `Victim Count`, `Suspect Count`, `Year`, `Month`, `Day`, `Hour`, `Police Station` |
| Suspect data | `Offense ID`, `Age`, `Gender`, `Barangay`, `Offense (Consolidated)` |
| Victim data | `Offense ID`, `Age`, `Gender`, `Barangay`, `Offense (Consolidated)` |

`Offense ID` is the join key across all three. Suspect and victim rows are one
person per row and get aggregated onto their incident.

`data/angeles_city_other_info.csv` is picked up automatically for barangay
population and area — you do not upload it.

**What happens when you submit.** The three files are cleaned (missing times and
coordinates imputed from similar incidents in the same barangay, ages of 0 read
as unrecorded, offense text mapped to the eight focus categories), joined,
merged with the existing dataset, re-aggregated into blocks, relabelled, and the
model is refit with five-fold randomised search. Expect one to two minutes.

On success you get record counts before and after, the new training shape, the
alarm distribution old versus new, and the recomputed thresholds. On failure the
error is shown and **the existing model is left untouched**.

**Two things to know before relying on this:**

- **The thresholds move.** Alarm levels are quantiles of the combined dataset,
  so adding data shifts the Low/Medium/High boundaries. A barangay can change
  level without its own incident count changing. Compare the old and new
  threshold figures in the success message.
- **Retraining is in-memory.** The refit model replaces the running one but the
  restored state does not survive a restart — `app.py` reloads
  `pulisai_xgb_model.joblib` from disk on boot.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| Every page empty, console says dataset not found | Run `python generate_data.py` |
| Predictions say "Model is not loaded" | Run `python xgboost_model.py` |
| Logged out after restarting the app | Normal without `PULISAI_SECRET_KEY` set |
| Map blank | `angeles_city_barangays.geojson` missing from the project root |
| Every selected barangay comes back Low with `N/A` fields | No historical record for that exact combination — try a broader one |
| Upload rejected | One of the three files missing, or a required column absent |
