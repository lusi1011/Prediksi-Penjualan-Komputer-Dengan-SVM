# -*- coding: utf-8 -*-
"""streamlit

Prediksi Penjualan Produk SuperStore dengan SVR (Streamlit Cloud Friendly)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# Fungsi bantu
# -----------------------------
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100

# -----------------------------
# Konfigurasi Halaman
# -----------------------------
st.set_page_config(page_title="Prediksi SVR SuperStore", layout="wide")
st.title("📊 Analisis Prediksi Penjualan Produk (SVR)")

# -----------------------------
# Upload File
# -----------------------------
uploaded_file = st.file_uploader("📂 Unggah file dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    st.subheader("🔍 Cuplikan Dataset")
    st.dataframe(df.head())

    # --- Pra-pemrosesan Data ---
    df_filtered = df[
        (df['Category'] == 'Technology') &
        (df['Sub-Category'] != 'Phones')
    ].copy()

    st.write(f"Jumlah data setelah filter: **{len(df_filtered)}** dari total **{len(df)}**")

    if df_filtered.empty:
        st.error("❌ Setelah filtering, data kosong. Tidak dapat melanjutkan.")
        st.stop()

    # Agregasi data per produk
    product_stats = df_filtered.groupby('Product Name').agg(
        Total_Quantity=('Quantity', 'sum'),
        Mean_Sales=('Sales', 'mean'),
        Mean_Profit=('Profit', 'mean'),
        Count_Orders=('Order ID', 'nunique')
    ).reset_index()

    product_stats = product_stats.replace([np.inf, -np.inf], np.nan).dropna()

    # --- Random Sampling Kecil (10 sampel acak) ---
    st.subheader("🧩 Sampling Data Acak")
    product_stats_sampled = product_stats.sample(n=10, random_state=42)
    st.write("📊 Menggunakan 10 data sampel acak untuk analisis:")
    st.dataframe(product_stats_sampled)

    # Gunakan hasil sampling untuk tahap selanjutnya
    X_products = product_stats_sampled[['Product Name', 'Mean_Sales', 'Mean_Profit', 'Count_Orders']]
    y_products = product_stats_sampled['Total_Quantity']
    feature_cols = ['Mean_Sales', 'Mean_Profit', 'Count_Orders']

    # Split data
    X_train_full, X_test_full, y_train, y_test = train_test_split(
        X_products, y_products, test_size=0.2, random_state=42
    )
    X_test = X_test_full[feature_cols]
    X_train = X_train_full[feature_cols]

    # Scaling
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()

    # Feature Selection
    selector = SelectKBest(score_func=f_regression, k=1)
    X_train_selected_all = selector.fit_transform(X_train_scaled, y_train_scaled)
    X_test_selected_all = selector.transform(X_test_scaled)
    selected_indices = selector.get_support(indices=True)
    selected_feature_names = [feature_cols[i] for i in selected_indices]
    selected_feature_name_for_plot = selected_feature_names[0]

    st.success("✅ Data berhasil diproses dan siap untuk pelatihan model.")

    # --- Hyperparameter Tuning ---
    with st.spinner("🔄 Melatih model SVR..."):
        param_grid_rbf = {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 'scale']}
        grid_search_rbf = GridSearchCV(SVR(kernel='rbf'), param_grid_rbf, cv=3, scoring='r2', n_jobs=-1)
        grid_search_rbf.fit(X_train_selected_all, y_train_scaled)
        best_rbf_svr = grid_search_rbf.best_estimator_

        param_grid_poly = {
            'C': [0.1, 1, 10],
            'gamma': [0.1, 1, 'scale'],
            'coef0': [0, 1, 2],
            'degree': [2, 3]
        }
        grid_search_poly = GridSearchCV(SVR(kernel='poly'), param_grid_poly, cv=3, scoring='r2', n_jobs=-1)
        grid_search_poly.fit(X_train_selected_all, y_train_scaled)
        best_poly_svr = grid_search_poly.best_estimator_

        param_grid_sigmoid = {
            'C': [0.1, 1, 10],
            'gamma': [0.1, 1, 'scale'],
            'coef0': [-2, -1, 0, 1, 2]
        }
        grid_search_sigmoid = GridSearchCV(SVR(kernel='sigmoid'), param_grid_sigmoid, cv=3, scoring='r2', n_jobs=-1)
        grid_search_sigmoid.fit(X_train_selected_all, y_train_scaled)
        best_sigmoid_svr = grid_search_sigmoid.best_estimator_

        model_linear = SVR(kernel='linear', C=10, gamma='scale').fit(X_train_selected_all, y_train_scaled)

    model_dict = {
        'Linear': model_linear,
        'Poly Tuned': best_poly_svr,
        'RBF Tuned': best_rbf_svr,
        'Sigmoid Tuned': best_sigmoid_svr
    }

    # --- Evaluasi Model ---
    results = []
    for name, model in model_dict.items():
        y_pred_scaled = model.predict(X_test_selected_all)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        results.append({'Model': name, 'R2': r2, 'MSE': mse, 'MAPE': mape})

    results_df = pd.DataFrame(results).sort_values(by='R2', ascending=False)
    st.subheader("📈 Hasil Evaluasi Model")
    st.dataframe(
        results_df.style.highlight_max(axis=0, subset=['R2'], color='lightgreen')
                         .highlight_min(axis=0, subset=['MSE', 'MAPE'], color='lightpink')
    )

    # --- Visualisasi ---
    st.subheader("🎨 Visualisasi Hasil Prediksi")

    X_test_original = scaler_X.inverse_transform(X_test_scaled)
    X_test_selected_original_plot = X_test_original[:, selected_indices[0]]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
    sns.set_style("whitegrid")

    for i, (name, model) in enumerate(model_dict.items()):
        ax = axes[i]

        x_min_orig = X_test_selected_original_plot.min()
        x_max_orig = X_test_selected_original_plot.max()
        x_smooth_orig = np.linspace(x_min_orig, x_max_orig, 300)

        selected_feature_index = selected_indices[0]
        mean_feat = scaler_X.mean_[selected_feature_index]
        std_feat = scaler_X.scale_[selected_feature_index]
        x_smooth_scaled = (x_smooth_orig - mean_feat) / std_feat
        x_smooth_scaled_reshaped = x_smooth_scaled.reshape(-1, 1)

        y_smooth_pred_scaled = model.predict(x_smooth_scaled_reshaped)
        y_smooth_pred_orig = scaler_y.inverse_transform(y_smooth_pred_scaled.reshape(-1, 1)).flatten()

        ax.scatter(X_test_selected_original_plot, y_test, color='gray', label='Aktual', alpha=0.6)
        ax.plot(x_smooth_orig, y_smooth_pred_orig, color='blue', linewidth=2, label=f'Prediksi {name}')

        ax.set_title(name)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlabel(selected_feature_name_for_plot)
        if i == 0:
            ax.set_ylabel('Total Quantity (Prediksi vs Aktual)')

    plt.tight_layout()
    st.pyplot(fig)

    st.success("✅ Semua proses selesai! Model SVR berhasil dijalankan dan divisualisasikan.")
else:
    st.info("👆 Unggah file CSV untuk memulai analisis.")
