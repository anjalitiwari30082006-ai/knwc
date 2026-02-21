import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier

# [cite_start]Page configuration [cite: 2]
[cite_start]st.set_page_config(page_title="Weather Classifier", layout="wide") [cite: 2]

# [cite_start]Title and description [cite: 2]
[cite_start]st.title("K-Nearest Neighbor - Weather Classification") [cite: 2]
[cite_start]st.markdown("Hello Everyone, so lets proceed.") [cite: 2]
[cite_start]st.markdown("This app uses K-Nearest Neighbors (KNN) from scikit-learn to classify weather conditions based on temperature and humidity levels.") [cite: 2]

# [cite_start]Training data [cite: 2]
X = np.array([
    [30, 70],
    [25, 80],
    [27, 60],
    [31, 65],
    [18, 75]
[cite_start]]) [cite: 2]

[cite_start]y = np.array([0, 1, 0, 0, 1]) # Label mapping: 0 = Sunny, 1 = Rainy [cite: 2]

# [cite_start]Label map [cite: 2]
[cite_start]label_map = {0: "Sunny", 1: "Rainy"} [cite: 2]

# [cite_start]Sidebar for user input [cite: 2]
[cite_start]st.sidebar.header("Input Parameters") [cite: 2]
[cite_start]temperature = st.sidebar.slider("Temperature (C)", min_value=20, max_value=35, value=26, step=1) [cite: 2]
[cite_start]humidity = st.sidebar.slider("Humidity (%)", min_value=50, max_value=90, value=78, step=1) [cite: 2]
[cite_start]n = st.sidebar.slider("KNN value", min_value=1, max_value=10, value=3, step=1) [cite: 2]

# [cite_start]Train the model [cite: 2]
[cite_start]knn = KNeighborsClassifier(n_neighbors=n) [cite: 2]
[cite_start]knn.fit(X, y) [cite: 2]

# [cite_start]Make prediction [cite: 2]
[cite_start]new_weather = np.array([[temperature, humidity]]) [cite: 2]
[cite_start]pred = knn.predict(new_weather)[0] [cite: 2]
[cite_start]pred_proba = knn.predict_proba(new_weather) [cite: 2]

# [cite_start]Display prediction result [cite: 2]
[cite_start]st.sidebar.subheader("Prediction Result") [cite: 2]
[cite_start]weather_label = label_map[pred] [cite: 2]
[cite_start]confidence = pred_proba[0][pred] * 100 [cite: 2]

if pred == 0:
    [cite_start]st.sidebar.success(f"Weather: {weather_label}") [cite: 2]
else:
    [cite_start]st.sidebar.info(f"Weather: {weather_label}") [cite: 2]

[cite_start]st.sidebar.metric("Confidence", f"{confidence:.1f}%") [cite: 2]

# [cite_start]Create visualization [cite: 2]
[cite_start]col1, col2 = st.columns(2) [cite: 2]

with col1:
    [cite_start]st.subheader("Classification Visualization") [cite: 2]
    [cite_start]fig, ax = plt.subplots(figsize=(8, 6)) [cite: 2]
    
    # [cite_start]Plot training data [cite: 2]
    [cite_start]ax.scatter(X[y == 0, 0], X[y == 0, 1], color="orange", label="Sunny", s=100, edgecolor="black", alpha=0.7) [cite: 2]
    [cite_start]ax.scatter(X[y == 1, 0], X[y == 1, 1], color="blue", label="Rainy", s=100, edgecolor="black", alpha=0.7) [cite: 2]
    
    # [cite_start]Plot prediction [cite: 2]
    [cite_start]colors = ["orange", "blue"] [cite: 2]
    [cite_start]ax.scatter(new_weather[0, 0], new_weather[0, 1], color=colors[pred], marker="*", s=300, edgecolor="black", label=f"New Day ({weather_label})", zorder=5) [cite: 2]
    
    # [cite_start]Formatting [cite: 2]
    [cite_start]ax.set_xlabel("Temperature (C)", fontsize=12, fontweight="bold") [cite: 2]
    [cite_start]ax.set_ylabel("Humidity (%)", fontsize=12, fontweight="bold") [cite: 2]
    [cite_start]ax.set_title("Weather Classification Model", fontsize=14, fontweight="bold") [cite: 2]
    [cite_start]ax.legend(fontsize=10) [cite: 2]
    [cite_start]ax.grid(True, alpha=0.3) [cite: 2]
    [cite_start]ax.set_xlim(20, 35) [cite: 2]
    [cite_start]ax.set_ylim(50, 90) [cite: 2]
    
    [cite_start]st.pyplot(fig) [cite: 2]
