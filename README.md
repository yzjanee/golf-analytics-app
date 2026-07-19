# ⛳ Golf Performance Analytics

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://golf-analytics-app-hg6jq6zwr2rvheo7ackfax.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-16a34a.svg)](LICENSE)

A machine learning web app that turns raw golf statistics into personalized practice priorities and score predictions. Built originally to help my high school varsity golf team prepare for the season — now open-sourced so any player or team can use it.

---

## Why I Built This

As a varsity golfer, I wanted to answer one question before every practice: **which part of my game is actually costing me the most strokes?**

Is it the fairways I'm missing? My iron approach distances? My putting patterns? Or my short game when I'm scrambling?

Instead of guessing, I built this app to run machine learning on round-by-round stats and produce a ranked practice plan backed by data — then extended it so my teammates could use it too.

---

## Features

### 📊 Overview
- KPI cards: avg score, recent trend, estimated Handicap Index, good-round %, consistency
- Score history chart with 3-round and 5-round rolling averages — markers color-coded by round type (tournament = blue ◆, practice = green ●)
- Season goal tracker: set a target handicap and see a progress bar + projection ("~12 more rounds at your current pace")
- Tournament vs. practice score breakdown
- Round-by-round data table with heatmap coloring

### 📈 Season Trends
- Handicap Index estimate that updates round-by-round
- Scoring consistency tracker (rolling standard deviation)
- Stat trends over the season vs. high school varsity field averages (FIR% 52, GIR% 55, Putts 32.5, U&D% 40)
- Estimated strokes gained/lost vs. field average per category

### 🎯 Statistical Analysis
- Pearson correlation of each stat with score — with readable interpretation
- Interactive correlation heatmap
- Scatter plots with OLS trendlines for any stat vs. score
- Good-round vs. bad-round comparison with grouped bar chart

### 🤖 ML Predictions
- Random Forest and XGBoost regression with Leave-One-Out CV (designed for small datasets)
- Actual vs. predicted scatter plot with best model
- Interactive predictor: adjust FIR%, GIR%, Putts, Up&Down% sliders → live score prediction + gauge
- Round classification (good/bad) accuracy scores

### 🔍 SHAP / Feature Importance
- Custom interactive Plotly SHAP beeswarm plot (each dot = one round, colored by stat value)
- Feature importance comparison across Random Forest, XGBoost, and SHAP methods
- Per-round SHAP values table — see exactly what drove every score

### 💡 Recommendations
- Gap analysis: current averages vs. good-round targets with animated progress bars
- Practice priorities auto-ranked by SHAP impact
- Specific drill suggestions for each stat category
- Auto-generated weekly practice schedule from your data
- Downloadable report (.txt) and data export (.csv)

### 👥 Team Mode
- Auto-activates when a `Player` column is in your CSV
- Team leaderboard with color-coded scoring averages
- Stat comparison with box plots
- Skill radar chart for multi-player comparison
- Score trends by player

### ➕ Live Round Logging
- Log new rounds directly in the sidebar — no CSV re-upload needed
- Supports all stats plus round type (Tournament / Practice)
- Appends to your dataset and reruns all charts instantly

---

## Quick Start

```bash
git clone https://github.com/yzjanee/golf-analytics-app.git
cd golf-analytics-app
pip install -r requirements.txt
streamlit run app.py
```

Then click **Individual Demo** or **Team Demo** on the landing page, or upload `sample_data.csv` to explore every feature with anonymous demo data.

---

## Data Format

| Column | Required | Description |
|--------|----------|-------------|
| `Round` | ✅ | Round number (1, 2, 3…) |
| `Score` | ✅ | Total strokes |
| `FIR%` | ✅ | Fairways in Regulation % |
| `GIR%` | ✅ | Greens in Regulation % (proxy for approach quality) |
| `Putts` | ✅ | Total putts |
| `Up&Down%` | ✅ | Scrambling / Up & Down % |
| `Date` | optional | YYYY-MM-DD |
| `Player` | optional | Name/initials — activates Team Mode |
| `Type` | optional | `Practice` or `Tournament` — enables type filter + breakdown |

**Privacy note:** `sample_data.csv` is fully anonymous demo data. No real teammate names, school names, or competition scores appear anywhere in this repo.

---

## Tech Stack

| | Tool |
|---|---|
| Web app | Streamlit |
| Data processing | pandas, numpy |
| Machine learning | scikit-learn (Random Forest, LOO-CV), XGBoost |
| Explainability | SHAP TreeExplainer |
| Visualizations | Plotly (all interactive), custom CSS/HTML |

---

## What I Built

- **Data pipeline** — structured round-by-round stats into a flexible schema that supports individual and team use; designed the CSV format to be easy for teammates to fill in
- **Exploratory analysis** — calculated Pearson correlations across all stats; found GIR% was the strongest predictor (r = −0.94) before running any ML
- **ML models** — implemented Leave-One-Out CV because standard k-fold is unreliable on small datasets; compared Random Forest and XGBoost for both regression (exact score) and classification (good/bad round)
- **SHAP integration** — replaced static matplotlib SHAP plots with a fully interactive Plotly beeswarm chart so teammates could explore their own rounds without needing to read code
- **Handicap tracker** — implemented simplified USGA Handicap Index formula that updates round-by-round so you can watch it drop over the season
- **Round log form** — built an in-app form so players can log new rounds directly without editing a CSV; data persists across the session and updates all charts live
- **Tournament vs. practice tracking** — added round type tagging so you can filter your analysis to just tournament rounds or see how your game differs between competitive and practice play
- **Season goal tracker** — set a target handicap and get a projected timeline based on your current improvement trajectory
- **Practice recommendation engine** — auto-ranks drills by SHAP importance and generates a weekly schedule from each player's data
- **Team mode** — extended the individual model to a multi-player dashboard (leaderboard, radar chart, box plots, score trends) so the whole team could see how they stacked up
- **UI/UX** — designed with custom CSS, a cohesive golf color palette, KPI cards, progress bars, and consistent chart styling

---

## Key Findings (from my data)

1. **GIR% is the #1 predictor** — 45.7% SHAP importance, r = −0.94 with score
2. **Scrambling matters more than most players think** — 25.9% importance; the gap between good and bad rounds was larger for Up&Down% than FIR%
3. **Putts are correlated but less causally important** — r = +0.94 but only 5.6% SHAP, because putts co-move with GIR% misses
4. **Good rounds require GIR% ≥ 56%** — in my data, every sub-76 round had at least 7 greens hit
5. **Tournament scores average ~1.8 strokes higher** than practice rounds — consistent with pressure and unfamiliar course conditions

---

## Project Structure

```
golf-analytics-app/
├── app.py                     # Main Streamlit app (~1,600 lines)
├── individualGolfModel.ipynb  # Original exploratory analysis notebook
├── sample_data.csv            # Anonymous 20-round demo dataset (with Type column)
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## License

MIT — fork it, adapt it, use it for your own team.
