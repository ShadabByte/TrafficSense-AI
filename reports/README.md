# 🚦 TrafficSense AI
### Intelligent Traffic Monitoring, Accident Analysis & Smart Route Recommendation System

<p align="center">
  <strong>An AI-powered traffic intelligence platform designed to explore traffic congestion, analyze accident severity, and support smarter urban mobility decisions.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python" />
  <img src="https://img.shields.io/badge/Streamlit-Application-red?logo=streamlit" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange?logo=scikitlearn" />
  <img src="https://img.shields.io/badge/Maps-OpenStreetMap-green?logo=openstreetmap" />
  <img src="https://img.shields.io/badge/Status-Active%20Development-brightgreen" />
</p>

---

## 📌 Overview

**TrafficSense AI** is a data science and machine learning project that brings multiple traffic intelligence capabilities together in one interactive application.

The platform explores traffic congestion patterns, predicts traffic conditions, analyzes accident severity, visualizes accident hotspots, and compares alternative routes using road-network information.

It also includes Bhopal-focused traffic analysis and a traffic data collection workflow.

The project combines Machine Learning, Data Analytics, Geospatial Visualization, and Interactive Dashboards to demonstrate how data-driven technologies can support smarter urban transportation.

## ✨ Key Features

### 🚗 Traffic Congestion Intelligence
- Traffic congestion classification.
- Traffic pattern exploration.
- Road-wise and time-based analysis.
- Interactive traffic visualizations.
- Machine learning-based congestion prediction.

### 🏙️ Bhopal Traffic Intelligence
- Bhopal-focused traffic estimation.
- Road network visualization using OpenStreetMap data.
- Traffic data collection workflow.
- Road-level traffic exploration.

### 🛣️ Smart Route Analyzer
- Road-network-based route analysis.
- Shortest-distance route comparison.
- Traffic-aware route recommendations.
- Configurable traffic impact scenarios.
- Alternative route visualization.

### 🚨 Accident Severity Analysis
- Machine learning-based accident severity prediction.
- Accident data exploration.
- Accident analytics dashboard.
- Accident hotspot visualization.
- Analysis of accident severity patterns.

### 📊 AI Traffic Dashboard
- Interactive data visualizations.
- Traffic intelligence summaries.
- Model evaluation information.
- Multiple analytical modules in one application.

### 🌐 Live Traffic Integration
- Optional TomTom traffic API integration.
- API credentials managed through local Streamlit secrets.
- Live traffic functionality depends on valid API access and connectivity.

---

## 🧠 Machine Learning

TrafficSense AI uses Python-based machine learning workflows to analyze traffic and accident-related data.

**Traffic congestion model**

The traffic classification pipeline uses:
- Road ID
- Hour
- Day

It predicts one of four congestion categories:

| Class | Meaning |
|---|---|
| FREE | Free-flowing traffic |
| LIGHT | Light congestion |
| MODERATE | Moderate congestion |
| HEAVY | Heavy congestion |

**Traffic model evaluation**

| Metric | Result |
|---|---:|
| Validation Accuracy | 80.72% |
| Balanced Accuracy | 72.35% |
| Macro F1-Score | 68.52% |

These are results from the project's recorded validation experiment, not a guarantee of real-world prediction performance.

**Accident severity model**

The accident analysis workflow also explores severity classification. The accident dataset has a significant class imbalance, so accuracy alone is not sufficient to judge model quality. Balanced accuracy and macro F1-score are important evaluation measures.

---

## 🗺️ Geospatial Technology

The project uses OpenStreetMap road-network information for Bhopal-focused mapping and route analysis.

Recorded road-network data:

| Information | Count |
|---|---:|
| Road segments | 10,062 |
| Road nodes | 4,203 |

These figures describe the processed road-network dataset used during development.

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Frontend / Dashboard | Streamlit |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Visualization | Matplotlib, Plotly |
| Maps | Folium, OpenStreetMap |
| Geospatial Analysis | GeoPandas |
| Network Analysis | NetworkX |
| API Integration | TomTom Traffic API |
| Development Tools | VS Code, Jupyter Notebook, Git, GitHub |

---

## 📂 Project Structure

```text
TrafficSense-AI/
│
├── app.py
├── START_TRAFFICSENSE.bat
│
├── app/
│   ├── dashboard.py
│   ├── traffic_page.py
│   ├── accident_page.py
│   └── map_page.py
│
├── api/
│   └── main.py
│
├── src/
│   ├── data_cleaning.py
│   ├── traffic_model.py
│   ├── accident_model.py
│   └── evalution.py
│
├── notebooks/
│   └── Data science notebooks
│
├── scripts/
│   └── Utility and evaluation scripts
│
├── reports/
│   └── Project reports
│
├── requirements.txt
├── .gitignore
└── README.md
```

*Some local datasets, trained model files, and configuration files are intentionally excluded from the public repository.*

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/ShadabByte/TrafficSense-AI.git
```

### 2. Open the project directory

```bash
cd TrafficSense-AI
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment

**Windows PowerShell:**

```powershell
.\venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Configure API credentials (optional)

For live traffic integration, configure your own API credentials using Streamlit's local secrets configuration.

Create:

```text
.streamlit/secrets.toml
```

Add the secret using the configuration key expected by the application's API integration.

**Important:** Never upload `secrets.toml`, API keys, passwords, or other credentials to GitHub.

### 7. Run the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal.

---

## 🔐 Data & Security

This repository intentionally excludes sensitive and local development assets.

- API credentials are not included.
- Private datasets are not included.
- Local trained model artifacts may need to be supplied separately.
- Environment and secret files should remain local.
- Users should configure their own API access where required.

Some modules may require the corresponding datasets and trained model files before they can run.

---

## ⚠️ Important Project Limitations

TrafficSense AI is an academic and portfolio-oriented prototype.

- Model performance depends on the quality and representativeness of available data.
- The traffic classification model was evaluated on the recorded dataset, not on a comprehensive live Bhopal traffic benchmark.
- Bhopal traffic estimation and route impact scenarios should not automatically be interpreted as verified real-time measurements.
- Live traffic features depend on API availability and valid credentials.
- Accident severity predictions are subject to dataset imbalance and model limitations.

The platform is intended for learning, experimentation, and demonstration—not as a replacement for official traffic or emergency services.

---

## 🚀 Future Scope

- Bhopal-specific traffic data collection at larger scale.
- Improved model validation using real-world observations.
- More reliable accident severity classification.
- Advanced traffic forecasting.
- Real-time traffic data integration.
- Improved alternative route recommendations.
- Cloud deployment and public access.
- Enhanced geospatial analytics.

---

## 🎯 Project Objective

The objective of TrafficSense AI is to demonstrate how machine learning, data analytics, and geospatial technologies can be combined to study urban traffic challenges and explore intelligent transportation solutions.

---

## 👨‍💻 Developer

**Shadab Khan**

BCA Student | Artificial Intelligence & Data Analytics

GitHub: [@ShadabByte](https://github.com/ShadabByte)

Project Repository: [TrafficSense AI](https://github.com/ShadabByte/TrafficSense-AI)

---

<p align="center">
  <strong>TrafficSense AI — Making Urban Mobility Smarter Through Data & AI.</strong>
</p>