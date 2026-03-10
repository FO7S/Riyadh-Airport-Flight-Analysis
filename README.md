# ✈️ Riyadh Airport Flight Analysis Dashboard

<p align="center">
  <a href="https://riyadh-airport-flight-analysis-pbt9nxlqkfjzkhehs68rzr.streamlit.app/">
    <img src="https://img.shields.io/badge/🚀_Live_Dashboard-Open_Streamlit_App-success?style=for-the-badge&logo=streamlit">
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-red?style=for-the-badge&logo=streamlit" />
  <img src="https://img.shields.io/badge/Python-Data%20Analysis-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/Plotly-Interactive%20Visuals-3f4f75?style=for-the-badge&logo=plotly" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Forecasting-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Status-Completed-success?style=for-the-badge" />
</p>

<p align="center">
An interactive aviation analytics dashboard designed to analyze flight departures from Riyadh Airport, explore traffic patterns, and generate business insights through interactive visualizations and short-term forecasting.
</p>

---

# 🚀 Live Dashboard

### 👉 Open the interactive dashboard

🔗 **https://riyadh-airport-flight-analysis-pbt9nxlqkfjzkhehs68rzr.streamlit.app/**

This dashboard allows users to explore Riyadh Airport flight activity interactively, analyze operational trends, and view forecasting results directly in the browser.

---

# 📌 Project Overview

This project presents a **complete data analytics workflow** applied to flight departures from **Riyadh Airport**.

The project transforms raw aviation data into an **interactive business intelligence dashboard** that helps explore airport operations, traffic patterns, airline activity, and destination trends.

The workflow includes multiple stages of the data science pipeline:

- Data preprocessing
- Exploratory Data Analysis (EDA)
- Business insight extraction
- Interactive dashboard development
- Short-term forecasting

The final result is an **interactive aviation analytics dashboard** designed for clear and intuitive data exploration.

---

# 🎯 Analytical Objectives

The project aims to answer important analytical questions such as:

- How does flight traffic change over time?
- What are the busiest operational hours?
- Which destinations receive the highest number of flights?
- Which airlines operate the largest share of departures?
- How are flights distributed across airport terminals?
- What patterns exist in weekly and monthly traffic trends?
- Can short-term airport activity be forecasted?

---

# 📊 Dashboard Features

The Streamlit dashboard provides several interactive analytical tools.

## Interactive Filters

Users can dynamically filter the dataset by:

- Date range
- Hour range
- Airline
- Terminal
- Destination search

These filters allow the dashboard to behave like a **real operational analytics tool**.

---

## Main Dashboard Components

### KPI Overview

Quick operational indicators including:

- Total flights
- Number of airlines
- Number of destinations
- Peak operational hour
- Busiest terminal

---

### Traffic Analysis

Visualizations that reveal airport activity patterns:

- Daily flight traffic trend
- Flights by hour of day
- Flights by day of week
- Monthly traffic distribution

---

### Destination Analysis

Route and connectivity insights including:

- Top international destinations from Riyadh
- Top domestic destinations within Saudi Arabia
- Top airlines by number of departures

---

### Forecasting Analysis

The dashboard also includes **short-term forecasting models** to estimate future airport traffic demand.

Features include:

- Chronological train/test split
- Model comparison
- Forecast error evaluation
- 14-day future flight prediction

---

# 🧠 Key Insights

Several insights emerge from the analysis:

- Flight activity remains **relatively stable** with normal operational fluctuations.
- Major regional hubs such as **Dubai, Cairo, and Istanbul** dominate international connectivity.
- A small number of airlines handle a **large proportion of departures**.
- Certain terminals experience **higher operational pressure**, indicating potential optimization opportunities.
- Forecasting suggests **short-term traffic demand is expected to remain stable**.

---

# 🔮 Forecasting Models Used

Two forecasting approaches were implemented:

### Linear Regression

A simple baseline model using time-based features.

### SARIMAX

A statistical time-series model capable of capturing seasonal patterns.

Model performance was evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

---

# 🛠️ Technologies Used

This project was built using:

**Programming & Data Analysis**

- Python
- Pandas
- NumPy

**Visualization**

- Plotly
- Streamlit

**Machine Learning**

- Scikit-learn
- Statsmodels

---

# 📂 Repository Structure

```
Riyadh-Airport-Flight-Analysis
│
├── app.py
├── requirements.txt
├── flights_RUH.csv
├── EDA FINAL_RUH.ipynb
└── README.md
```

---

# 📊 Dataset Description

The dataset contains operational records for flight departures from **Riyadh Airport**.

Key attributes include:

- Flight number
- Airline information
- Destination airport
- Terminal
- Scheduled departure time
- Aircraft information

The dataset enables analysis of traffic behavior and airline operations.

---

# 👨‍💻 Author

**Faisal Al-Sulami**

Data Science Graduate  
Focused on data analytics, machine learning, and interactive data products.

---

# ⭐ Project Highlights

✔ End-to-end data analytics workflow  
✔ Interactive Streamlit dashboard  
✔ Aviation traffic insights  
✔ Forecasting models for demand prediction  
✔ Professional portfolio project

---

💡 If you found this project useful, feel free to **star ⭐ the repository**.
