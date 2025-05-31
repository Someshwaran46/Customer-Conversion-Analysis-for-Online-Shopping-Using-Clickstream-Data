# 🛒 Customer Conversion Analysis for Online Shopping Using Clickstream Data  

## 🚀 Project Overview  
This project focuses on **analyzing clickstream data** to enhance customer engagement and optimize e-commerce sales. A **Streamlit web application** is developed to predict customer conversion, estimate potential revenue, and segment users for personalized marketing.  

## 🔍 Business Use Cases  
✔ **Customer Conversion Prediction** – Identify potential buyers to improve marketing efficiency.  
✔ **Revenue Forecasting** – Predict user spending to optimize pricing strategies.  
✔ **Customer Segmentation** – Group users based on browsing behavior for personalization.  
✔ **Churn Reduction** – Detect potential cart abandoners and enable proactive re-engagement.  
✔ **Product Recommendations** – Suggest relevant products based on user interactions.  

## 📂 Folder Structure
```bash
├── Main/
│   ├── 1. EDA.ipynb                    # Exploratory Data Analysis: Understand distributions, outliers, trends.
│   ├── 2. Data Preprocessing.ipynb     # Data cleaning, encoding, scaling, and feature engineering.
│   ├── 3. Classification.ipynb         # Model training and evaluation for classification tasks (e.g., purchase prediction).
│   ├── 4. Regression.ipynb             # Model training and evaluation for regression tasks (e.g., revenue prediction).
│   └── 5. Clustering.ipynb             # Unsupervised learning for customer segmentation using clustering algorithms.
│
├── Pickled Data/
│   ├── chart_data.pkl                  # Preprocessed visualization data for faster loading/rendering.
│   ├── decision_tree_model.pkl         # Trained Decision Tree Regression model (used in `Regression.ipynb` or app).
│   ├── gradient_boostl.pkl             # Trained Gradient Boosting Classifier model (used in `Classification.ipynb` or app).
│   ├── kmeans.pkl                      # Trained KMeans model used for customer segmentation.
│   ├── pca.pkl                         # PCA transformation model used for dimensionality reduction.
│   ├── preprocessing_pipeline.pkl      # Complete preprocessing pipeline including transformers used for inference.
│   └── scaler.pkl                      # Scaler (e.g., StandardScaler or MinMaxScaler) for transforming numerical data.
│
├── Source data/
│   ├── preprocessed.csv                # Final cleaned and preprocessed dataset ready for modeling.
│   ├── test_data.csv                   # Data used for evaluating model performance after training.
│   ├── train_data.csv                  # Data used for training ML models.
│   └── uploaded_file_bulk.csv          # CSV used for bulk prediction uploads through the app interface.
│
├── venv/                               # Python virtual environment (typically excluded from version control).
│
└── app.py                              # Main Streamlit app for user interaction, prediction, and visualization.
```
## 🧠 Machine Learning Approach  
**1️⃣ Data Preprocessing & Feature Engineering**  
- Handling missing values, encoding categorical variables, scaling numerical features.  
- Session analysis, behavior tracking, and clickstream pattern extraction.  

**2️⃣ Model Building & Evaluation**  
- **Classification:** Logistic Regression, Random Forest, XGBoost, Neural Networks.  
- **Regression:** Linear Regression, Ridge, Lasso, Gradient Boosting.  
- **Clustering:** K-Means, DBSCAN, Hierarchical Clustering.  
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, RMSE, Silhouette Score.  

**3️⃣ Streamlit Web App Development**  
- **Real-time predictions** for conversion (classification) and revenue estimation (regression).  
- **Customer segmentation visualization** for targeted marketing strategies.  
- **Interactive UI** for data upload, visualization, and insights generation.  

## 🛠 Tech Stack  
🔹 Python, Pandas, NumPy, Scikit-learn, XGBoost, TensorFlow  
🔹 Matplotlib, Seaborn for **EDA & Visualizations**  
🔹 Streamlit for **web application & model deployment**  

## Getting Started

### Follow these steps to clone the repository and run the app locally using Streamlit.
### Clone the Repository
```bash
git clone https://github.com/Someshwaran46/Customer-Conversion-Analysis-for-Online-Shopping-Using-Clickstream-Data.git
cd Customer-Conversion-Analysis-for-Online-Shopping-Using-Clickstream-Data
```
### Create a virtual environment and activate it
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```
### Install required dependencies
```bash
pip install -r requirements.txt
```
### Running the App
Run the Streamlit app with:
```bash
streamlit run app.py
```
---
## 🎯 Explore. Predict. Segment. Succeed.

Interactive Web Application:
-  Build a Streamlit interface that allows users to upload CSV files or input values manually.
  
Key Features:
- Real-time predictions for conversion (classification).
- Revenue estimation (regression).
- Display customer segments (clustering visualization).
- Show visualizations like bar charts, pie charts, and histograms.
---
## 📬 Feedback

Feel free to open issues or submit pull requests! Improvements, and suggestions are always welcome 🙌.
For clarifications drop an email to somesh4602@gmail.com.

