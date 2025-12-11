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

#### Page 1 : Forcast
<img width="1701" height="804" alt="Image" src="https://github.com/user-attachments/assets/fd15ac20-8437-4b58-b0e5-826c4f6e5c1b" />

#### Page 2 : Operations
<img width="1701" height="807" alt="Image" src="https://github.com/user-attachments/assets/2db82f53-e60b-4664-9743-2167fa2c5dba" />

> One can download and intract with Dashboard or can access the live Dashboard.

## 🎯 Key Achievements 

### ✔ Gather information from multiple data sources  
Loaded **airports**, **airlines**, and **flights** datasets and integrated them via Python + Power BI.

### ✔ Identify trends, patterns, and business implications  
Analyzed delay causes, peak congestion hours, and risk drivers.

### ✔ Automate data processing  
Python scripts automatically clean, aggregate, and export BI-ready datasets.

### ✔ Produce and track KPIs  
Created cancellation rate, delay metrics, risk scores, and route-level KPIs.

### ✔ Build dashboards & interactive visualizations  
Developed a professional Power BI dashboard with drilldowns, maps, and risk heatmaps.

### ✔ Predictive modeling  
Implemented a Random Forest delay prediction model with explainability.

### ✔ Present insights to business stakeholders  
Dashboard tells a complete story from **operations → risk → forecasting**.

## 📌 Installation & Setup

### **1. Clone the repository**
```bash
git clone https://github.com/your-username/airline-analytics.git
cd airline-analytics

python -m venv venv          # Create a Virtual Environment
venv\Scripts\activate        # Windows
# or
source venv/bin/activate     # macOS/Linux

```

### **2. Install Python dependencies**
```bash
pip install -r requirements.txt
```

### **3. Run preprocessing**
```bash
python analytics_export.py
```

### **4. Train model**
```bash
python train_risk_model.py
```

### **5. Generate predictions**
```bash
python model_predict_full.py
```

### **6. Run the Python Dashboard**
Launch the Streamlit dashboard:
```bash
streamlit run python/app.py
```

### **7. Open Power BI Dashboard**
Load:

```bash
/PowerBI/Airline_Analyst_dash.pbix
```
> Inside Power BI → click Refresh to load the newest processed datasets and risk scores.


## 🧩 Key Challenges & How I Solved Them

### ⚠️ 1. Extremely Large Dataset (4M+ flight records)
Working with DOT flight data required processing millions of rows, which caused performance bottlenecks in both Python and Power BI.

#### ✅ Solution
- **Chunked Loading in Python:**  
  Instead of loading the entire dataset into memory, I used pandas chunk processing to stream and clean data in batches.
- **Pre-aggregated Fact Tables:**  
  Created `Aggregated_Route`, `Aggregated_Carrier`, and `aggregated_metrics` tables to reduce data volume before modeling.
- **Optimized Data Types:**  
  Converted categorical/string columns to category types and reduced datetime precision where appropriate.

These steps **reduced memory usage by ~80%** and enabled smooth processing and visualization.

---

### ⚠️ 2. Many-to-Many Relationships (ROUTE & CARRIER level)
The original dataset contained repeating route codes and carrier combinations, causing Power BI’s automatic relationship detection to fail.

#### ✅ Solution
- **Created a DimRoute Dimension Table:**  
  Extracted unique ROUTE values into a dedicated dimension table, enforcing a clean *1-to-many* relationship.
- **Implemented a Proper Star Schema:**  
  Designed the model with:
  - `clean_flights_for_bi` as the **Fact Table**
  - Route, Carrier, and Risk Score tables as **Dimensions**
- **Removed Ambiguous Cross Filters:**  
  Ensured filters flow *downward* from dimensions → fact table to avoid inconsistent aggregations.

This eliminated ambiguity and enabled stable filtering for the dashboard.

---

### ⚠️ 3. Memory Issues During Model Training
The Random Forest model initially failed with memory errors when training on 4M rows.

#### ✅ Solution
- **Feature Pruning:**  
  Removed non-informative or redundant columns, reducing dimensionality.
- **Efficient Encodings:**  
  Used `OrdinalEncoder` + frequency encoding instead of one-hot encoding to dramatically reduce the feature space.
- **Cardinality Reduction:**  
  Grouped rare routes and carriers under “Other” categories to stabilize training.

These optimizations allowed the model to train successfully on large, structured aviation datasets.

---

### ⚠️ 4. High Cardinality in ROUTE & AIRPORT Data
ROUTE was a combination of ORIGIN–DEST pairs, resulting in thousands of unique keys.

#### ✅ Solution
- Created **route frequency metrics** to encode importance.
- Applied **domain knowledge** (airport hubs, major carriers) to build meaningful aggregations.
- Integrated **airport latitude/longitude** to enable geospatial Power BI visuals.

This ensured both machine learning and BI layers worked efficiently with a large categorical space.

---

### ⚠️ 5. Integrating Python Pipeline with Power BI
Power BI required clean, pre-processed CSVs, but the raw data had missing airport codes, inconsistent carrier labels, and mixed datatypes.

#### ✅ Solution
- Built a **fully automated Python pipeline** that outputs BI-ready datasets:
  - Cleaned flight data
  - Risk scores
  - Aggregated route/carrier metrics
- Included validation rules to ensure:
  - No duplicate dimension keys
  - Consistent column names
  - Stable refresh cycles

This made Power BI refreshes **deterministic, fast, and repeatable**.

---

## 🤝 Contributing

Contributions are welcome!  
If you’d like to improve the data pipeline, enhance the Power BI dashboard, optimize the ML model, or add new analytics modules, feel free to collaborate.

### 🐞 Issues & Feature Requests

If you encounter any bugs, have questions, or want to request a new enhancement:

👉 Create an issue here:
[https://github.com/your-username/airline-analytics/issues](https://github.com/ansh90378/-AirlineOps_Analytics-Forecasting/issues)

## 📬 Contact
Ansh

📧 Email: ansh90378@gmail.com
🔗 LinkedIn: https://www.linkedin.com/in/ansh-chauhan-4430741a9

## 📄 License
MIT License — Free to use, modify, and share.
