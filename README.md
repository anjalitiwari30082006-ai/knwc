import numpy as np sanjaliTiwari

import matplotlib.pyplot as plt

import streamlitas at

From sklearn.neighbors import KNeighborsClassifier

erage configuration

st.set page.config(page_title "Weather Classifier", layout-"wide")

Title and description

st.titln("K-Nearest-Neighbor Weather Classification")

st.markdown Hello Everyone, se lets proceed.")

st.markdown

this app uses K-Nearest Neighbors(KNN) from scikit-learn to classify weather conditions

based on temperature and humidity levels.

13

Training data

np.array([ [30, 70), [31.65]. [23, 551.

[25, 801. [27,601,

【28,751

упр.аттау([0, 1, 0, 0, 1, 132

Label mapping

label map

0: "Sunny",

1: "Ratny"

Sidabar Tuer usar input

st-sidebar.header(" Input Parameters")

temperature st.sidebar.slider("Temperature (°C)", min value 20, max value 35, value 26, step=1)

humidity st.sidebar.slider("Humidity (%)", min valoe 50, max value 90, value 78, step 1)

Train the model using scikit-learn's KheighborsClassifier

ST.sidebar.slider("KNN value", min value, max valuesto, value, step=12

knn KheighborsClassifier(n_neighborson)

knn.fit(x, y)

Mave prediction

now weather np.array([[temperature, humidity]1)

pred knn.predict(new_weather)[0]

pred proba kon.predict proba(new weather) [ 0]

Display prediction rasult

st.sidebar.subheader(" Prediction Result")

st.sidebar.markdown("

wwather Label label map[pred]

confidence pred proba[pred] 100

Color based on prediction

st.sidebar.successfweather: (weather label)

else:

LT pred 0:

st.sidebar.info(f**weather (weather labelj

st.sidebar.metric Confidence", f"(confidence: 1fys")

Mais content Create visualization

colt, col2 st.columns(2)

colt:

st.subheader(" Classification Visualization")

plt-subplats(figsize (8,633

Plst training data

ax.scatter(x[y=0, 0], x[y=0, 1], color="orange", label="Sunny", s-100, edgecolor", alpha-0.7) ax.scatter(x[y-1, 0), xy-1, 1), color-"blue", label-"Rainy", s-100, edgecolor-k", alpha-0.7)

Fiat new prediction

colors ["orange", "blue"]

color-colors[pred), marker, s-300, edgecolor "black",

ax.scatter(new weather[0, 0], new weather [0, 13.

label-f New day: (weather label)", zorder-s)

x.sat slabel("Temperature (°C)", fontsize 12, fontweight"bold")

ax.set ylabel("Humidity (%)", fontsize 12, fontweight "bold")

set title("Weather Classification Model, fontsize-14, fontweight "hold">

ax.lugund(fontsize=10)

ax.grid(True, alpha=0.3)

ax.set slim(20, 35)

ax.set ylim(50, 90)

st.pyplot(fig)