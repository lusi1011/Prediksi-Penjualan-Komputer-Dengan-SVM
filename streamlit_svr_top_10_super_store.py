# -*- coding: utf-8 -*-
"""streamlit

Prediksi Penjualan Produk SuperStore dengan SVM (Streamlit Cloud Friendly)
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
st.set_page_config(page_title="Prediksi SuperStore Melalui SVM", layout="wide")
st.title("Analisis Prediksi Penjualan Produk (SVM)")

# -----------------------------
# Pengumpulan Data
# -----------------------------
uploaded_file = "SuperStore_Sales_Dataset.csv"

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Kutipan Dataset Penjualan Komponen Komputer")
    st.dataframe(df.head(10))

# -----------------------------
# Pemrosesan Awal Data
# -----------------------------
    # Penyaringan dan Pembersihan Data
    df_filtered = df[
        (df['Category'] == 'Technology') &
        (df['Sub-Category'] != 'Phones')
    ].copy()

    # (Bersihkan nilai-nilai yang tidak terisi)
    df_filtered = df_filtered.dropna(axis=1, how='all')
    df_filtered.rename(columns={'Row ID+O6G3A1:R6': 'Row ID'}, inplace=True)
    df_filtered['Returns'] = df_filtered['Returns'].fillna(0)
    df_filtered.columns = df_filtered.columns.str.strip()

    st.subheader("Kutipan Dataset Setelah Disaring")
    st.dataframe(df_filtered.head(10))
    st.write(f"Jumlah data setelah disaring: **{len(df_filtered)}** dari total **{len(df)}**")

    if df_filtered.empty:
        st.error("Data berubah menjadi kosong. Tidak mampu dilanjutkan.")
        st.stop()

    # Integrasi Data Setiap Komponen
    product_stats = df_filtered.groupby('Product Name').agg(
        Total_Quantity=('Quantity', 'sum'),
        Mean_Sales=('Sales', 'mean'),
        Mean_Profit=('Profit', 'mean'),
        Count_Orders=('Order ID', 'nunique')
    ).reset_index()


# -----------------------------
# Penentuan Sampel & Fitur
# -----------------------------
X_products = product_stats[['Mean_Sales', 'Mean_Profit', 'Count_Orders']]
y_products = product_stats['Total_Quantity']
feature_cols = ['Mean_Sales', 'Mean_Profit', 'Count_Orders']

# Pemisahan Data
X_train, X_test, y_train, y_test = train_test_split(
    X_products, y_products, test_size=1, random_state=42
)

# Simpan salinan X_test sebelum scaling untuk visualisasi akhir
X_test_orig_df = X_test.copy()
# Ambil Product Name untuk tabel TOP 10
test_indices = X_test.index
product_names_test = product_stats.loc[test_indices, 'Product Name'].values

# -----------------------------
# Standarisasi & Seleksi Fitur
# -----------------------------
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()

# Cari 1 fitur terbaik untuk visualisasi regresi 2D nanti
selector = SelectKBest(score_func=f_regression, k=1)
selector.fit(X_train_scaled, y_train_scaled)
selected_idx = selector.get_support(indices=True)[0]
selected_feature_name = feature_cols[selected_idx]

# Gunakan SEMUA fitur untuk pelatihan model (X_train_scaled), 
# namun untuk plot regresi di bawah, kita akan gunakan subset fitur terbaik.
X_train_sel = X_train_scaled[:, [selected_idx]]
X_test_sel = X_test_scaled[:, [selected_idx]]

# -----------------------------
# Training Model (Grid Search)
# -----------------------------
with st.spinner("Melatih model SVM dengan Grid Search..."):
    # RBF
    gs_rbf = GridSearchCV(SVR(kernel='rbf'), {'C': [1, 10], 'gamma': [0.1, 1]}, cv=3).fit(X_train_sel, y_train_scaled)
    # Poly
    gs_poly = GridSearchCV(SVR(kernel='poly'), {'C': [1, 10], 'degree': [2, 3]}, cv=3).fit(X_train_sel, y_train_scaled)
    # Sigmoid
    gs_sig = GridSearchCV(SVR(kernel='sigmoid'), {'C': [1, 10], 'gamma': [0.1, 1]}, cv=3).fit(X_train_sel, y_train_scaled)
    # Linear
    model_lin = SVR(kernel='linear', C=1).fit(X_train_sel, y_train_scaled)

model_dict = {
    'Linear': model_lin,
    'Poly_Tuned': gs_poly.best_estimator_,
    'RBF_Tuned': gs_rbf.best_estimator_,
    'Sigmoid_Tuned': gs_sig.best_estimator_
}

# -----------------------------
# Visualisasi Matriks Korelasi
# -----------------------------
st.markdown("---")
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Matriks Korelasi")
    fig_corr, ax_corr = plt.subplots()
    corr = product_stats[['Total_Quantity', 'Mean_Sales', 'Mean_Profit', 'Count_Orders']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
    st.pyplot(fig_corr)

# -----------------------------
# Evaluasi Kinerja
# -----------------------------
results = []
pred_results = {}

for name, model in model_dict.items():
    y_pred_scaled = model.predict(X_test_sel)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    results.append({
        'Model': name,
        'R2': r2_score(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'MAPE': mean_absolute_percentage_error(y_test, y_pred)
    })
    pred_results[name] = y_pred

results_df = pd.DataFrame(results).sort_values(by='R2', ascending=False)
with col2:
    st.subheader("Hasil Kinerja Kernel SVM")
    st.dataframe(results_df.style.highlight_max(subset=['R2'], color='#90ee90'))

# -----------------------------
# TOP 10 Analisis
# -----------------------------
st.markdown("---")
st.subheader("🏆 Top 10 Produk Berdasarkan Prediksi")
tabs = st.tabs(list(model_dict.keys()))

for i, name in enumerate(model_dict.keys()):
    with tabs[i]:
        top_df = pd.DataFrame({
            'Nama Produk': product_names_test,
            'Aktual': y_test.values,
            'Prediksi': pred_results[name],
            'Bulat Bawah': np.floor(pred_results[name])
        }).sort_values(by='Prediksi', ascending=False).head(10)
        st.table(top_df)

# -----------------------------
# Visualisasi Kurva Regresi
# -----------------------------
st.markdown("---")
st.subheader(f"Visualisasi Regresi (Fitur Terbaik: {selected_feature_name})")

fig_reg, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
x_plot_scaled = np.linspace(X_test_sel.min(), X_test_sel.max(), 100).reshape(-1, 1)
x_plot_orig = (x_plot_scaled * scaler_X.scale_[selected_idx]) + scaler_X.mean_[selected_idx]

for i, (name, model) in enumerate(model_dict.items()):
    y_plot_scaled = model.predict(x_plot_scaled)
    y_plot_orig = scaler_y.inverse_transform(y_plot_scaled.reshape(-1, 1)).flatten()
    
    axes[i].scatter(X_test_orig_df[selected_feature_name], y_test, color='red', alpha=0.5, label='Actual')
    axes[i].plot(x_plot_orig, y_plot_orig, color='black', lw=2, label='Prediction')
    axes[i].set_title(name)
    axes[i].set_xlabel(selected_feature_name)
    if i == 0: axes[i].set_ylabel("Total Quantity")
    axes[i].legend()

st.pyplot(fig_reg)
st.success("Analisis Selesai!")
