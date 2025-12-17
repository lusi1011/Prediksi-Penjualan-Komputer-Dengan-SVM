# ================================
# STREAMLIT APP - TOP 10 BARANG TERLARIS (SVR)
# Dataset: SuperStore_Sales_Dataset.csv
# Platform: Google Colab + Streamlit
# ================================

# --- 1. IMPORT LIBRARY ---
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline

# --- 2. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Analisis TOP 10 Barang Terlaris", layout="wide")

st.title("🏆 Analisis TOP 10 Barang Terlaris")
st.write("Tampilan 10 produk terlaris berdasarkan prediksi Quantity menggunakan kernel regresi SVM.")

# --- 3. LOAD DATASET ---
@st.cache_data
def load_data():
    df = pd.read_csv("SuperStore_Sales_Dataset.csv")
    return df

df = load_data()

# --- 4. PREPROCESSING ---
# Encode Product Name
le = LabelEncoder()
df['Product_Encoded'] = le.fit_transform(df['Product Name'])

X = df[['Product_Encoded', 'Sales', 'Discount', 'Profit']]
y = df['Quantity']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- 5. FUNGSI TRAIN SVR ---
def train_svr(kernel_type):
    if kernel_type == "linear":
        param_grid = {'svr__C': [0.1, 1, 10]}
        svr = SVR(kernel='linear')

    elif kernel_type == "rbf":
        param_grid = {
            'svr__C': [1, 10],
            'svr__gamma': [0.1, 1]
        }
        svr = SVR(kernel='rbf')

    elif kernel_type == "poly":
        param_grid = {
            'svr__C': [1, 10],
            'svr__degree': [2, 3]
        }
        svr = SVR(kernel='poly')

    else:  # sigmoid
        param_grid = {
            'svr__C': [1, 10],
            'svr__gamma': [0.1, 1]
        }
        svr = SVR(kernel='sigmoid')

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', svr)
    ])

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring='neg_mean_absolute_error'
    )

    grid.fit(X_train, y_train)
    return grid.best_estimator_

# --- 6. TAB INTERFACE ---
tabs = st.tabs(["Linear", "Poly Tuned", "RBF Tuned", "Sigmoid Tuned"])

kernels = ["linear", "poly", "rbf", "sigmoid"]

for tab, kernel in zip(tabs, kernels):
    with tab:
        st.subheader(f"SVR {kernel.capitalize()} Kernel (Top 10)")

        model = train_svr(kernel)
        df['Prediksi'] = model.predict(X)
        df['Bulat Bawah'] = np.floor(df['Prediksi']).astype(int)

        top10 = (
            df.groupby('Product Name')
            .agg({
                'Quantity': 'sum',
                'Prediksi': 'mean',
                'Bulat Bawah': 'mean'
            })
            .reset_index()
            .sort_values('Prediksi', ascending=False)
            .head(10)
        )

        top10.columns = ['Nama Produk', 'Aktual', 'Prediksi', 'Bulat Bawah']
        st.dataframe(top10, use_container_width=True)

# --- 7. FOOTER ---
st.caption("Model: Support Vector Regression | Dataset: SuperStore Sales")
