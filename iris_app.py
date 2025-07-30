# iris_app.py

import streamlit as st
import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Configure the page
st.set_page_config(page_title="Iris Classifier", layout="centered")

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Train the KNN model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Streamlit App UI
st.title("🌸 Iris Flower Classifier - AI Web App")
st.markdown("Enter the flower measurements below to predict the species of the flower.")

# Sliders for input
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Predict button
if st.button("Predict Flower Type"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Display input for confirmation
    st.write("🔍 You entered:")
    st.write(f"Sepal Length: {sepal_length} cm")
    st.write(f"Sepal Width: {sepal_width} cm")
    st.write(f"Petal Length: {petal_length} cm")
    st.write(f"Petal Width: {petal_width} cm")

    # Predict the class
    prediction = model.predict(input_data)
    predicted_species = target_names[prediction[0]]

    # Display result
    st.success(f"🌼 Predicted Species: **{predicted_species.capitalize()}**")

# Sidebar: Model information
accuracy = accuracy_score(y_test, model.predict(X_test))
st.sidebar.header("📊 Model Info")
st.sidebar.write(f"Model Used: **K-Nearest Neighbors (k=3)**")
st.sidebar.write(f"Accuracy on Test Set: **{accuracy * 100:.2f}%**")
st.sidebar.write("Data: scikit-learn's Iris Dataset")
