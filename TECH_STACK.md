# House Price Predictor Pro — Tech Stack & Architecture

This document outlines the detailed technology stack, frameworks, algorithms, and how they are utilized within the **House Price Predictor Pro** full-stack machine learning web application.

---

## 🏗️ 1. Architecture Overview

The system follows a classic decoupled **Client-Server Architecture**:
- **Frontend (Client):** A static HTML/CSS/JS single-page application that handles the user interface and data visualization.
- **Backend (Server):** A Python REST API that serves machine learning predictions and model evaluation metrics.
- **Machine Learning Pipeline:** An offline training system that preprocesses data, trains models, and saves the best model bundle (serialized via `joblib`) for the backend to use.

---

## 🛠️ 2. Core Python Backend & ML Stack

The backend handles all heavy lifting, including data wrangling, model training, and API serving.

### Frameworks & Libraries
*   **Python (v3.11):** The core programming language.
*   **Flask (v3.0.3):** A lightweight WSGI web application framework used to build the REST API.
    *   *Usage:* Exposes `/predict`, `/metrics`, and `/features` endpoints. Translates HTTP requests containing house details into numpy arrays for the ML model.
*   **Flask-CORS:** Adds Cross-Origin Resource Sharing headers.
    *   *Usage:* Allows the frontend (running locally or on a different domain) to securely call the Flask API.
*   **Pandas & NumPy:** Standard data manipulation libraries.
    *   *Usage:* Used for reading the CSV dataset, handling data frames, feature engineering (vectorized operations), and feeding arrays into scikit-learn.
*   **Scikit-Learn (v1.4.2):** The primary machine learning library.
    *   *Usage:* Provides implementations for preprocessing techniques (Imputation, Scaling, PCA) and all predictive algorithms (Linear models, Ensembles, KNN).
*   **Joblib:** A set of tools to provide lightweight pipelining in Python.
    *   *Usage:* Serializes (saves to disk) the trained model and the entire preprocessing pipeline into `model.pkl` so it can be rapidly loaded into memory when the Flask server starts.
*   **Matplotlib & Seaborn:** Data visualization libraries.
    *   *Usage:* Automatically generates model evaluation plots (heatmaps, scatter plots, bar charts) during the training phase, which are then served to the frontend.

---

## 🧠 3. Machine Learning Algorithms

The project trains an ensemble of six different algorithms, optimizing them using `GridSearchCV` (cross-validation) to find the best approach.

### 1. Preprocessing Algorithms
Before training, the data goes through a rigid pipeline:
*   **SimpleImputer (Median Strategy):** Handles missing data by replacing blank values with the median of that column. Robust to extreme outliers.
*   **MinMaxScaler:** Normalizes all numeric features to a scale between 0 and 1. This prevents features with large numbers (like `LotArea`) from overwhelming distance-based algorithms like KNN.
*   **Principal Component Analysis (PCA):** A dimensionality reduction technique.
    *   *Usage:* Extracts the most important mathematical variance from the data. 

### 2. Predictive Algorithms (Regression)
*   **Linear Regression:** Fits a straight line that minimizes the mean squared error between predicted and target values. Acts as a performance baseline.
*   **Ridge Regression (L2 Regularization):** Similar to Linear Regression, but adds a penalty for large coefficients. Prevents overfitting.
*   **Lasso Regression (L1 Regularization):** Shrinks less important feature coefficients to exactly zero, essentially performing automatic feature selection.
*   **K-Nearest Neighbors (KNN Regressor):** Predicts the price based on the average price of the 'K' most similar houses in the dataset.
*   **Random Forest Regressor:** An ensemble learning method that constructs multitudes of decision trees during training and outputs the average prediction of individual trees. Highly accurate and robust to overfitting.
    *   *Side Usage:* Extracted specifically to calculate "Feature Importance" (e.g., proving `OverallQual` impacts price the most).
*   **Gradient Boosting Regressor:** *[Winning Model]* Sequentially builds decision trees, where each new tree specifically tries to correct the errors (residuals) made by the previous trees. Yields the highest R² score (~0.88).

---

## 🎨 4. Frontend Web Stack

The frontend is a static page emphasizing modern "Stitch-style" design aesthetics, specifically avoiding bloated JS frameworks (like React/Next.js) since it is a single view dashboard.

### Frameworks & Libraries
*   **HTML5:** Semantic structuring of the application (Hero, Predictor, Analytics, Docs).
*   **Tailwind CSS (via CDN):** A utility-first CSS framework.
    *   *Usage:* Used for rapid UI development. Provides the responsive grid system, spacing, typography, and dark-mode color palettes without writing custom CSS files for layout.
*   **Vanilla CSS3 (`style.css`):**
    *   *Usage:* Layered on top of Tailwind to achieve complex visual effects like Glassmorphism (blur filters), animated ambient background orbs, synchronized CSS animations, and custom scrollbars. 
*   **Vanilla JavaScript (ES6+):**
    *   *Usage:* Handles DOM manipulation, intercepts form submissions, validates input parameters, and manages asynchronous `fetch()` calls to the Flask API.
*   **Chart.js:** An HTML5-based JavaScript charting library.
    *   *Usage:* Used to render the dynamic "Feature Importance" horizontal bar chart and the interactive "Model R² Score" comparison chart directly in the browser.
*   **Google Fonts:** `Inter` for highly legible UI elements, and `Space Grotesk` for striking, geometric headers.

---

## 🔄 5. Data Flow Summary

1.  **Training Phase (Offline):** 
    *   `train.csv` is loaded -> Imputed -> Scaled -> PCA reduced.
    *   6 algorithms are trained inside a 5-fold cross-validation grid.
    *   Best model (`Gradient Boosting`) and Pipeline are saved to `model.pkl`.
    *   Stats are saved to `metrics.json`.
2.  **Serving Phase (Online):** 
    *   Flask API starts and loads `model.pkl` into RAM.
3.  **Client Request:** 
    *   User fills UI form -> JS sends JSON to `POST /predict`.
    *   Flask passes raw data to `Pipeline.transform()` -> passes transformed data to `GradientBoosting.predict()`.
    *   Flask calculates confidence score and returns Price + INR Category.
    *   Frontend JavaScript updates the DOM with a smooth counting animation.
