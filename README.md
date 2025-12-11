# ✈️ Airline Operations Analytics & Risk Forecasting  
### End-to-End Data Engineering + Analytics + BI Dashboard Project  

*A complete real-world data analysis pipeline built to demonstrate data modeling, automation, visualization, and predictive analytics expertise.*

---

## 🔥 Project Overview (Why This Project Matters)

This project is a **full-cycle data analytics solution** built on real U.S. airline delay data. It tackles a realistic business question:

> How can we understand operational delays, forecast future risks, and provide insights that guide business decisions?

This project is an end-to-end airline analytics system that processes U.S. flight delay data, trains a machine learning model to predict route-level delay risk, and visualizes insights in a live interactive dashboard (Python) and a rich Power BI report.

It demonstrates real-world data engineering, predictive modeling, and BI development workflows used inside airline operations teams (OCC, Network Planning, Crew Scheduling). 

The solution answers critical industry questions:

- Which routes and airlines are most delay-prone?

- When do delays spike during the day?

- How can we predict route-level delay risk using ML?

- How can analysts use dashboards for daily operations?

---

## 🚀 2. What This Project Can Do

### 🔹 Data Engineering

- Import and clean raw FAA flight delay dataset, Merge airlines & airport metadata

- Generate engineered features (ROUTE, DEP_HOUR, IS_DELAY, etc.) and Produce analytics-ready datasets for BI

### 🔹 Machine Learning

- Train a route-level delay risk model (risk_model.pkl)

- Predict risk scores using full processed dataset

### 🔹 Interactive Dashboards

#### 🟡 Python Dashboard (app.py)

- Live delay statistics

- Risk prediction visualization

- Route/city-level mapping

- Interactive filtering

#### 🟠 Power BI Dashboard

- Rich operations KPIs

- Route-level and airline-level drilldowns

- Heatmaps, slicers, geographic visuals

- Route line mapping using lat/long

- Delay trend analysis

- ML-powered risk interpretation

---

## 🛠 Tech Stack

| Area                   | Tools Used                                                                             | Description                                           |
|------------------------|----------------------------------------------------------------------------------------|-------------------------------------------------------|
| Data Processing        | Python (Pandas, NumPy)                                                                 | Data Cleaning & Validation
| Automation             | Python scripts, modular ML pipeline                                                    | Calculating and generating important matrices.
| Visualization          | Power BI (DAX, data modeling, maps, Python visuals, KPIs), Python (Matlibplot, plotly) | Data Modeling and Dashboarding.
| Machine Learning       | scikit-learn, xgboost (Random Forest for delay risk scoring)                           | Predicting insights
| Version Control        | Git, GitHub                                                                            | Managing new versions on project.

---

## 📁 Repository Structure

> Folder names may vary slightly in your repo. Adjust as needed.

```bash
├── data
│   ├── raw_data
│   │   ├── flights.csv           # Kaggle / DOT flight delay data (2015)
│   │   ├── airlines.csv          # Airline metadata
│   │   └── airports.csv          # Airport metadata with latitude/longitude
│   ├── processed_data
│       ├── clean_flights_for_bi.csv
│       ├── processed_delays.csv
│       ├── aggregated_metrics.csv
│       ├── risk_input_candidates.csv
│       └── risk_scores.csv
│
├── python
│   ├── analytics_export.py       # Cleans & prepares BI-ready datasets
│   ├── train_risk_model.py       # Trains ML risk model
│   ├── model_predict_full.py     # Scores full dataset with trained model
│   ├── app.py                    # Interactive Python/Streamlit dashboard
│   ├── risk_model.pkl            # Saved trained model
│   └── risk_feature_map.json     # Feature configuration / metadata
│
├── PowerBI
│   └── Airline_Analyst_dash.pbix # Power BI report file
│
└── README.md
```

# 🚀 Project Workflow (Step-by-Step Execution)

### 1️⃣ Load & Prepare Raw Data (Kaggle DOT Dataset)
Raw dataset includes:
- flights → On-time performance, cancellations, delays
- airports → Full location information including latitude & longitude
- airlines → Carrier metadata

This dataset is ideal because it matches real industry reporting used by:
U.S. Department of Transportation’s Bureau of Transportation Statistics.

### 2️⃣ Generate Delay Risk Predictions
Run:

python analytics_export.py

Produces:
risk_scores.csv → probability of delay for each route, carrier, origin, and destination.

### 3️⃣ Train Predictive Delay Risk Model
Run:

python train_risk_model.py

This script:
- Trains a XGBOOST
-  Computes feature importance

Generates risk_model.pkl

### 4️⃣ Run model_predict_full.py
 to Clean & Transform Data

This script performs:
- Data cleaning & null handling
- Delay/cancellation calculations
- Feature engineering (route, carrier metrics, frequencies)
- Creation of BI-ready tables
- Export of final datasets including risk_input_candidates

Outputs include:
- clean_flights_for_bi.csv
- aggregated_metrics.csv
- aggregated_route.csv
- aggregated_carrier.csv

risk_input_candidates.csv

### 5️⃣ Build the Interactive Power BI Dashboard
The Power BI dashboard displays:

⭐ Key KPIs
- On-Time Score
- Cancellation Rate
- Total Flights
- Avg Risk Score
- High-Risk Route Count
  
⭐ Operational Insights
- Delay patterns by hour, airline, route
- Route-level flight flows plotted on the US map
- Heatmap of Carrier × Route risk combinations
- Forecasted risk vs actual delays

⭐ Predictive Insights
- Probability of future delays
- Comparison of high-risk vs low-risk routes
  
> “Create dashboards, graphs, and visualizations to showcase business performance and provide sector benchmarking.

## 📊 Dashboard Preview

<img width="1701" height="804" alt="image" src="https://github.com/user-attachments/assets/87b0751b-7eae-4141-ab1a-ae19ee68004d" />

