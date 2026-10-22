# DataForge AI — TinyFish + ML prototype

A local Streamlit MVP for the project concept: **natural-language request → live web → validated dataset → ML model → evidence**.

## What works in this prototype

- TinyFish Search to discover public web pages.
- TinyFish Fetch to retrieve page content before extraction.
- TinyFish Agent to extract structured laptop listings from relevant pages.
- Normalization, basic price validation, deduplication, provenance (`source_url`, `source_title`, `collected_at`) and CSV persistence.
- Ridge Regression baseline compared with Extra Trees.
- Hold-out MAE, RMSE and R²; 5-fold CV MAE when enough data is available.
- A Streamlit interface to collect, inspect, train, evaluate and predict.

**Important limitations:** this is a hackathon prototype, not a general autonomous ML platform. Search results may not be listing pages; sites may block access; Agent extraction can return incomplete or incorrect fields. Review the CSV before trusting results. Prices are asking prices, not confirmed sale prices. The code intentionally does not fabricate missing data. The document's 300–500 rows are a target, not guaranteed by one collection run.

## 1. Install prerequisites on Arch Linux

Open a terminal and install Python, pip, and Git:

```bash
sudo pacman -Syu
sudo pacman -S --needed python python-pip git
```

Make a project directory and virtual environment:

```bash
mkdir -p ~/projects
cd ~/projects
# If you downloaded and extracted the project zip, cd into its dataforge_ai folder.
cd dataforge_ai
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Arch Linux uses an externally managed system Python. **Do not use `sudo pip install`**; use the project's `.venv`.

## 2. Configure TinyFish securely

1. Open the [TinyFish API Keys page](https://agent.tinyfish.ai/api-keys).
2. Create a key if needed.
3. Copy `.env.example` to `.env` and edit it:

```bash
cp .env.example .env
nano .env
```

Put your key after `TINYFISH_API_KEY=`. Do not add quotes unless your key requires them. Save and exit.

Lock down the file:

```bash
chmod 600 .env
```

If you previously pasted an API key into a chat, terminal screenshot, public repo, or shared document, revoke/rotate it in TinyFish and use a fresh key.

## 3. Launch the app

From the project root, with `.venv` active:

```bash
streamlit run app.py
```

Streamlit should print a local URL, usually `http://localhost:8501`. Open it in your browser.

## 4. First end-to-end run

1. Open **Collect live data**.
2. Use the default query or enter a narrower query such as `used gaming laptop RTX 3050 India price`.
3. Start with 2–5 result pages to control run time and API usage.
4. Inspect the collection notes. Some search results will not be product listing pages; that's expected.
5. Open **Inspect dataset**. Check prices, specs, and source URLs; delete bad rows from `data/laptop_listings.csv` if necessary.
6. Aim for at least 8 valid rows for the initial evaluation. More varied listings are needed for meaningful metrics; the concept document's target is 300–500 rows.
7. Open **Train & evaluate** and click **Train and evaluate**.
8. Open **Predict** and enter a laptop's specs. The estimate is produced by the saved ML model, not directly by TinyFish.

## 5. Project layout

```text
dataforge_ai/
├── app.py               # Streamlit UI
├── data_pipeline.py     # TinyFish search/fetch/agent + validation
├── ml.py                # preprocessing, training, metrics, prediction
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── data/                # created automatically; CSV is local
└── models/              # created when training
```

## 6. Troubleshooting

### `ModuleNotFoundError: No module named 'tinyfish'`

Check the virtual environment is active, then:

```bash
python -m pip install -r requirements.txt
```

### TinyFish says API key missing / 401

Check `.env` exists in the same directory from which you run Streamlit, the variable is named exactly `TINYFISH_API_KEY`, and restart Streamlit after editing `.env`.

### No usable listings

Try a more specific query, use fewer known marketplace/category URLs, or manually inspect the search results. Collection quality depends on live sites and TinyFish results. Do not count fabricated/demo rows as real evidence.

### Model training says fewer than 8 rows

Collect more real listings and review their `price_inr` values. Do not lower the validation threshold just to make the demo appear successful.

### `pip` reports an externally managed environment

Activate `.venv` and use `python -m pip`, without `sudo`.

## 7. TinyFish API references

- [TinyFish Quick Start](https://docs.tinyfish.ai/quick-start)
- [TinyFish Search examples](https://docs.tinyfish.ai/search-api/examples)
- [TinyFish documentation](https://docs.tinyfish.ai/)

## Next upgrades

- Add a dedicated schema-planning LLM step for arbitrary user prompts.
- Implement source-specific collection recipes and pagination for 2–3 pre-tested marketplaces.
- Add robust GPU/CPU normalization, outlier review and listing-level similarity matching.
- Add XGBoost/LightGBM as optional model candidates once the dataset is large enough.
- Add tests and manual-review status for every extracted row.
