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
    st.dataframe(df.head())

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
    st.dataframe(df_filtered.head())
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
# Penentuan Sampel Data
# -----------------------------
    st.subheader("Random Sampling Data")
    product_stats_sampled = product_stats.sample(sample_size=100, random_state=42)
    st.write("Contoh sampel untuk analisis:")
    st.dataframe(product_stats_sampled)

    X_products = product_stats_sampled[['Product Name', 'Mean_Sales', 'Mean_Profit', 'Count_Orders']]
    y_products = product_stats_sampled['Total_Quantity']
    feature_cols = ['Mean_Sales', 'Mean_Profit', 'Count_Orders']


# -----------------------------
# Pemisahan Data
# -----------------------------
    X_train_full, X_test_full, y_train, y_test = train_test_split(
        X_products, y_products, test_size=0.5, random_state=42
    )
    X_test = X_test_full[feature_cols]
    X_train = X_train_full[feature_cols]


# -----------------------------
# Standarisasi Data
# -----------------------------
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()


# -----------------------------
# Penentuan Fitur Paling Efektif
# -----------------------------
    selector = SelectKBest(score_func=f_regression, k=1)
    X_train_selected_all = selector.fit_transform(X_train_scaled, y_train_scaled)
    X_test_selected_all = selector.transform(X_test_scaled)
    selected_indices = selector.get_support(indices=True)
    selected_feature_names = [feature_cols[i] for i in selected_indices]
    selected_feature_name_for_plot = selected_feature_names[0]

    st.success("Data berhasil diproses dan siap untuk pelatihan model.")


# -----------------------------
# Implementasi Parameter Kernel SVM Dengan Uji Paling Efektif
# -----------------------------
    with st.spinner("Sedang melatih model regresi SVM..."):
        param_grid_rbf = {
            'C': [0.1, 1, 10],
            'gamma': [0.1, 1]
        }
        grid_search_rbf = GridSearchCV(SVR(kernel='rbf'), param_grid_rbf, cv=3, scoring='r2', n_jobs=-1)
        grid_search_rbf.fit(X_train_selected_all, y_train_scaled)
        rbf_param = grid_search_rbf.best_estimator_

        param_grid_poly = {
            'C': [0.1, 1, 10],
            'gamma': [0.1, 1, 10],
            'coef0': [0, 1, 2],
            'degree': [2, 3]
        }
        grid_search_poly = GridSearchCV(SVR(kernel='poly'), param_grid_poly, cv=3, scoring='r2', n_jobs=-1)
        grid_search_poly.fit(X_train_selected_all, y_train_scaled)
        poly_param = grid_search_poly.best_estimator_

        param_grid_sigmoid = {
            'C': [0.1, 1, 10],
            'gamma': [0.1, 1, 10],
            'coef0': [-2, -1, 0, 1, 2]
        }
        grid_search_sigmoid = GridSearchCV(SVR(kernel='sigmoid'), param_grid_sigmoid, cv=3, scoring='r2', n_jobs=-1)
        grid_search_sigmoid.fit(X_train_selected_all, y_train_scaled)
        sigmoid_param = grid_search_sigmoid.best_estimator_

        linear_param = SVR(kernel='linear', C=10, gamma='scale').fit(X_train_selected_all, y_train_scaled)

    model_dict = {
        'Linear': linear_param,
        'Poly_Tuned': poly_param,
        'RBF_Tuned': rbf_param,
        'Sigmoid_Tuned': sigmoid_param
    }

# -----------------------------
# Visualisasi Model
# -----------------------------    
    # Matriks Korelasi
    corr = product_stats[['Total_Quantity', 'Mean_Sales', 'Mean_Profit', 'Count_Orders']].corr()

    fig, ax = plt.subplots(figsize=(4,4))
    im = ax.imshow(corr, cmap='coolwarm')  # Menggunakan colormap yang lebih menarik

    # Tambahkan Label Kolom dan Baris
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.columns)))
    ax.set_xticklabels(corr.columns)
    ax.set_yticklabels(corr.columns)

    # Rotasi Label X
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=8)

    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            value = corr.iloc[i, j]
            text_color = "white" if abs(value) > 0.6 else "black"
            ax.text(j, i, round(value, 2), ha='center', va='center', color=text_color, fontsize=6)

    # Judul dan Colorbar, serta Tampilan Plot Matriks
    st.markdown("---")
    ax.set_title("Matriks Korelasi", pad=20, fontsize=12)
    colorbar = plt.colorbar(im, ax=ax, label='Koefisien Korelasi', shrink=0.5)
    colorbar.set_label('Koefisien Korelasi', fontsize=8)

    st.pyplot(fig)

    # Hasil Kinerja dari Kernel SVM
    results = []
    predictions_df = pd.DataFrame({'Actual': y_test})
    for name, model in model_dict.items():
        y_pred_scaled = model.predict(X_test_selected_all)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        predictions_df[f'Predicted_{name}'] = y_pred
        results.append({'Model': name, 'R2': r2, 'MSE': mse, 'MAPE': mape})

    results_df = pd.DataFrame(results).sort_values(by='R2', ascending=False)
    st.markdown("---")
    st.subheader("Hasil Kinerja dari Kernel SVM")
    st.dataframe(
        results_df.style.highlight_max(axis=0, subset=['R2'], color='lightgreen')
                         .highlight_min(axis=0, subset=['MSE', 'MAPE'], color='lightgreen')
    )

    st.write("**Keterangan Bagi Kernel Nonlinear:**")
    st.write(f"**Parameter Poly Tuned:** {poly_param}")
    st.write(f"**Parameter RBF Tuned:** {rbf_param}")
    st.write(f"**Parameter Sigmoid Tuned:** {sigmoid_param}")

    # Analisis TOP 10 Barang Terlaris
    # Ambil Nama Produk dan Integrasikan dengan Prediksi
    best_sellers_df = X_test_full['Product Name'].reset_index(drop=True).to_frame()
    best_sellers_df['Actual_Quantity'] = y_test.values
    best_sellers_df['Predicted_Linear'] = predictions_df['Predicted_Linear'].values
    best_sellers_df['Predicted_Poly_Tuned'] = predictions_df['Predicted_Poly_Tuned'].values
    best_sellers_df['Predicted_RBF_Tuned'] = predictions_df['Predicted_RBF_Tuned'].values
    best_sellers_df['Predicted_Sigmoid_Tuned'] = predictions_df['Predicted_Sigmoid_Tuned'].values

    best_sellers_df['Floor_Linear'] = np.floor(best_sellers_df['Predicted_Linear'])
    best_sellers_df['Floor_Poly'] = np.floor(best_sellers_df['Predicted_Poly_Tuned'])
    best_sellers_df['Floor_RBF'] = np.floor(best_sellers_df['Predicted_RBF_Tuned'])
    best_sellers_df['Floor_Sigmoid'] = np.floor(best_sellers_df['Predicted_Sigmoid_Tuned'])

    st.markdown("---")
    st.title("🏆 Analisis TOP 10 Barang Terlaris")
    st.markdown("Tampilan **10 Produk terlaris** berdasarkan prediksi Quantity melalui kernel regresi SVM.")

    tab1, tab2, tab3, tab4 = st.tabs(["Linear", "Poly Tuned", "RBF Tuned", "Sigmoid Tuned"])
    TOP_N = 10

    def show_top_n(df, pred_col, floor_col, kernel_name):
        top_n_df = df.sort_values(by=pred_col, ascending=False).head(TOP_N)
        display_df = top_n_df[['Product Name', 'Actual_Quantity', pred_col, floor_col]].copy()
        display_df.columns = ['Nama Produk', 'Aktual', 'Prediksi', 'Bulat Bawah']
        st.subheader(f"SVR {kernel_name} Kernel (Top {TOP_N})")
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Actual": st.column_config.NumberColumn(format="%d"),       
                "Prediksi": st.column_config.NumberColumn(format="%.2f"),   
                "Bulat Bawah": st.column_config.NumberColumn(format="%d"),  
            }
        )

    with tab1:
        show_top_n(best_sellers_df, 'Predicted_Linear', 'Floor_Linear', 'Linear')
    with tab2:
        show_top_n(best_sellers_df, 'Predicted_Poly_Tuned', 'Floor_Poly', 'Poly Tuned')
    with tab3:
        show_top_n(best_sellers_df, 'Predicted_RBF_Tuned', 'Floor_RBF', 'RBF Tuned')
    with tab4:
        show_top_n(best_sellers_df, 'Predicted_Sigmoid_Tuned', 'Floor_Sigmoid', 'Sigmoid Tuned')

    # Visualisasi Hasil Kinerja dari Kernel SVM
    st.markdown("---")
    st.subheader("Visualisasi Hasil Kinerja dari Kernel SVM")

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

        ax.scatter(X_test_selected_original_plot, y_test, color='red', label='Aktual', alpha=0.6)
        ax.plot(x_smooth_orig, y_smooth_pred_orig, color='black', linewidth=2, label=f'Prediksi {name}')

        ax.set_title(name)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlabel(selected_feature_name_for_plot)
        if i == 0:
            ax.set_ylabel('Total Quantity (Prediksi vs Aktual)')

    plt.tight_layout()
    st.pyplot(fig)

    st.success("Model Regresi SVM berhasil diproses dan divisualisasikan.")
