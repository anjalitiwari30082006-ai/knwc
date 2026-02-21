import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier

# Page configuration
st.set_page_config(page_title="Weather Classifier", layout="wide")

# Title and description
st.title("K-Nearest Neighbor - Weather Classification")
st.markdown("Hello Everyone, so lets proceed.")
st.mardown("""
_This app uses K-Nearest Neighbors (KNN) from scikit-learn to classify weather conditions 
based on temperature and humidity levels._
""")

# Training data
X = np.array([
    [30, 70],
    [25, 80],
    [27, 60],
    [31, 65],
    [18, 75]
])

y = np.array([0, 1, 0, 0, 1]) 

# Label map
label_map = {
     0: "Sunny", 
     1: "Rainy"
}


# Sidebar for user input
st.sidebar.header("Input Parameters")
temperature = st.sidebar.slider("Temperature (C)", min_value=20, max_value=35, value=26, step=1)
humidity = st.sidebar.slider("Humidity (%)", min_value=50, max_value=90, value=78, step=1)

# Train the model 
n = st.sidebar.slider("KNN value", min_value=1, max_value=10, value=3, step=1)
knn = KNeighborsClassifier(n_neighbors=n)
knn.fit(X, y)

# Make prediction
new_weather = np.array([[temperature, humidity]])
pred = knn.predict(new_weather)[0]
pred_proba = knn.predict_proba(new_weather)

# Display prediction result
st.sidebar.markdown("---")
st.sidebar.subheader("Prediction Result")
weather_label = label_map[pred]
confidence = pred_proba[0][pred] * 100

# Color based on prediction
if pred == 0:
    st.sidebar.success(f"Weather: {weather_label}** "")
else:
    st.sidebar.info(f"Weather: {weather_label}** ")

st.sidebar.metric("Confidence", f"{confidence:.1f}%")

# Main content - Create visualization
col1, col2 = st.columns(2)

with col1:
    st.subheader("Classification Visualization")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot training data
    ax.scatter(X[y == 0, 0], X[y == 0, 1], color="orange", label="Sunny", s=100, edgecolor="black", alpha=0.7)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], color="blue", label="Rainy", s=100, edgecolor="black", alpha=0.7)
    
    # Plot prediction
    colors = ["orange", "blue"]
    ax.scatter(new_weather[0, 0], new_weather[0, 1], 
              color=colors[pred], marker="*", s=300, edgecolor="black", 
              label=f"New Day ({weather_label})", zorder=5)
    
    ax.set_xlabel("Temperature (C)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Humidity (%)", fontsize=12, fontweight="bold")
    ax.set_title("Weather Classification Model", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(20, 35)
    ax.set_ylim(50, 90)
    
    st.pyplot(fig)
