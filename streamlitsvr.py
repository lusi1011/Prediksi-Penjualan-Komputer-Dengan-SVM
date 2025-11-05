import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# Konfigurasi Halaman
st.set_page_config(layout="wide", page_title="SVR Sales Analysis")
st.title("Analisis Support Vector Regression (SVR) dengan Seleksi Fitur")
st.caption("Menganalisis dan memprediksi 'Quantity' berdasarkan fitur yang dipilih ('Sales' or 'Profit').")

# --- 1. Fungsi Pembantu dan Pemrosesan Data ---

def mean_absolute_percentage_error(y_true, y_pred):
    """Menghitung MAPE, aman dari pembagian nol."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_true[y_true == 0] = 1e-6
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

@st.cache_data
def load_and_process_data():
    """Memuat, memfilter, dan membersihkan data dari CSV."""
    try:
        df = pd.read_csv('SuperStore_Sales_Dataset.csv')
        df = df.rename(columns={'Row ID+O6G3A1:R6': 'Row ID'})
    except FileNotFoundError:
        st.error("File 'SuperStore_Sales_Dataset.csv' tidak ditemukan. Analisis dibatalkan.")
        return None, None
    except Exception as e:
        st.error(f"Error saat membaca CSV: {e}")
        return None, None
        
    # Filter data (Technology, non-Phones)
    df_tech = df[df['Category'] == 'Technology']
    df_com = df_tech[df_tech['Sub-Category'] != 'Phones']
        
    # Ambil fitur yang relevan
    df_clean = df_com[['Sales', 'Profit', 'Quantity']].dropna().copy()
    
    # Filter data ekstrem
    df_clean = df_clean[
        (df_clean['Sales'] < 3000) & (df_clean['Sales'] > 0) &
        (df_clean['Profit'] > -500) & (df_clean['Profit'] < 800)
    ]
    
    # Sampling yang Aman
    n_samples = 250
    if len(df_clean) == 0:
        st.warning("Tidak ada data yang tersisa setelah filter.")
        return None, None
        
    if len(df_clean) < n_samples:
        n_samples = len(df_clean)
        
    return df_clean.sample(n=n_samples, random_state=42), df_com

@st.cache_data
def run_feature_selection(df_clean):
    """Menjalankan SelectKBest untuk menemukan fitur terbaik."""
    if df_clean is None or len(df_clean) < 10:
        return pd.DataFrame()

    features = ['Sales', 'Profit']
    target = 'Quantity'
    
    X = df_clean[features]
    Y = df_clean[target]
    
    # Scaling diperlukan agar F-scores dapat dibandingkan
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Gunakan f_regression untuk regresi, k='all' untuk mendapatkan skor semua fitur
    selector = SelectKBest(f_regression, k='all')
    selector.fit(X_scaled, Y)
    
    scores_df = pd.DataFrame({
        'Fitur': features,
        'F-Score': selector.scores_
    }).sort_values(by='F-Score', ascending=False)
    
    return scores_df

@st.cache_data(show_spinner="Menjalankan SVR untuk semua kernel...")
def run_all_svr_analysis(df_clean, selected_feature):
    """Menjalankan SVR untuk semua kernel pada fitur yang dipilih."""
    if df_clean is None or len(df_clean) < 10:
        return pd.DataFrame(), pd.DataFrame()

    # PERBAIKAN: X sekarang dinamis berdasarkan selected_feature
    X = df_clean[selected_feature].values.reshape(-1, 1)
    Y = df_clean['Quantity'].values.reshape(-1, 1)

    test_split_size = 0.3
    if len(df_clean) * test_split_size < 1:
        test_split_size = 0.1 
            
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_split_size, random_state=42)
    
    if len(X_test) == 0:
        return pd.DataFrame(), pd.DataFrame()

    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    Y_train_scaled = scaler_Y.fit_transform(Y_train).ravel()

    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    model_results = []
    
    X_range_scaled = np.linspace(X_train_scaled.min(), X_train_scaled.max(), 50).reshape(-1, 1)
    plot_data_list = []
    
    # PERBAIKAN: DataFrame plot menggunakan nama fitur yang dinamis
    n_plot_samples = 50
    df_test_actual = pd.DataFrame({selected_feature: X_test.ravel(), 'Quantity': Y_test.ravel()})
    if len(df_test_actual) > n_plot_samples:
        df_test_actual = df_test_actual.sample(n=n_plot_samples, random_state=42)
    df_test_actual['Type'] = 'Actual'
    
    for kernel in kernels:
        # Konfigurasi parameter default
        if kernel == 'linear': svr = SVR(kernel=kernel, C=10)
        elif kernel == 'poly': svr = SVR(kernel=kernel, C=50, degree=2, gamma='auto')
        elif kernel == 'rbf': svr = SVR(kernel=kernel, C=50, gamma='auto')
        elif kernel == 'sigmoid': svr = SVR(kernel=kernel, C=1, gamma=0.1, coef0=0)
            
        svr.fit(X_train_scaled, Y_train_scaled)
        
        Y_pred_test_scaled = svr.predict(X_test_scaled)
        Y_pred_test = scaler_Y.inverse_transform(Y_pred_test_scaled.reshape(-1, 1))
        
        mse = mean_squared_error(Y_test, Y_pred_test)
        r2 = r2_score(Y_test, Y_pred_test)
        mape = mean_absolute_percentage_error(Y_test, Y_pred_test)
        model_results.append({'Kernel': kernel, 'MSE': mse, 'R2': r2, 'MAPE': mape})

        Y_pred_range_scaled = svr.predict(X_range_scaled)
        Y_pred_range = scaler_Y.inverse_transform(Y_pred_range_scaled.reshape(-1, 1))

        # PERBAIKAN: DataFrame prediksi menggunakan nama fitur yang dinamis
        df_pred = pd.DataFrame({
            selected_feature: scaler_X.inverse_transform(X_range_scaled).ravel(),
            'Quantity': Y_pred_range.ravel(),
            'Kernel': kernel,
            'Type': 'Prediction'
        })
        
        df_actual_temp = df_test_actual.copy()
        df_actual_temp['Kernel'] = kernel
        
        plot_data_list.append(df_actual_temp)
        plot_data_list.append(df_pred)

    df_metrics = pd.DataFrame(model_results)
    df_plot_final = pd.concat(plot_data_list, ignore_index=True)

    return df_metrics, df_plot_final

def create_svr_chart(df_data, kernel_name, title, selected_feature, x_label):
    """Membuat chart Altair individual, kini dengan sumbu-X dinamis."""
    df_kernel = df_data[df_data['Kernel'] == kernel_name].copy()
    if df_kernel.empty:
        return None 

    base = alt.Chart(df_kernel).encode(
        # PERBAIKAN: Sumbu X dan tooltip sekarang dinamis
        x=alt.X(selected_feature, title=x_label, scale=alt.Scale(zero=False)),
        y=alt.Y('Quantity', title='Jumlah (Quantity)', scale=alt.Scale(zero=False)),
        tooltip=[selected_feature, 'Quantity']
    )
    
    scatter = base.transform_filter(
        alt.FieldEqualPredicate(field='Type', equal='Actual')
    ).mark_point(opacity=0.6, size=50, color='gray', filled=True)
    
    line = base.transform_filter(
        alt.FieldEqualPredicate(field='Type', equal='Prediction')
    ).mark_line(size=3, color='#FF4B4B')
    
    chart = (scatter + line).properties(title=title).interactive()
    return chart

@st.cache_data(show_spinner="Menghitung SVR Kustom...")
def run_custom_svr(df_clean, selected_feature, kernel, C, gamma, degree=3, coef0=0.0):
    """Menjalankan SVR kustom pada fitur yang dipilih."""
    X = df_clean[selected_feature].values.reshape(-1, 1)
    Y = df_clean['Quantity'].values.reshape(-1, 1)

    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    Y_scaled = scaler_Y.fit_transform(Y).ravel()
    
    # Konfigurasi SVR
    if kernel == 'poly':
        svr = SVR(kernel=kernel, C=C, degree=degree, gamma=gamma)
    elif kernel == 'sigmoid':
        svr = SVR(kernel=kernel, C=C, gamma=gamma, coef0=coef0)
    else:
        svr = SVR(kernel=kernel, C=C, gamma=gamma)
        
    svr.fit(X_scaled, Y_scaled)
    
    X_range_scaled = np.linspace(X_scaled.min(), X_scaled.max(), 100).reshape(-1, 1)
    Y_pred_range_scaled = svr.predict(X_range_scaled)
    Y_pred_range = scaler_Y.inverse_transform(Y_pred_range_scaled.reshape(-1, 1))

    # PERBAIKAN: DataFrame plot menggunakan nama fitur yang dinamis
    df_pred_custom = pd.DataFrame({
        selected_feature: scaler_X.inverse_transform(X_range_scaled).ravel(),
        'Quantity': Y_pred_range.ravel(),
        'Kernel': kernel,
        'Type': 'Prediction'
    })
    
    df_actual_custom = pd.DataFrame({selected_feature: X.ravel(), 'Quantity': Y.ravel()})
    df_actual_custom['Kernel'] = kernel
    df_actual_custom['Type'] = 'Actual'
    
    df_plot_custom = pd.concat([df_actual_custom, df_pred_custom], ignore_index=True)
    return df_plot_custom

# --- 2. Main Streamlit Execution ---
df_clean, df_raw_filtered = load_and_process_data()

# Tampilkan Data Mentah
if df_raw_filtered is not None:
    with st.expander("Lihat Data Mentah Setelah Filter (Kategori: Technology, Sub-Kategori: Bukan Phones)"):
        st.write(df_raw_filtered)

# Hanya jalankan jika data berhasil di-load dan diproses
if df_clean is not None and len(df_clean) >= 10:
    
    # --- BARU: Bagian 1: Analisis Seleksi Fitur ---
    st.header("1. Analisis Seleksi Fitur (Otomatis)")
    st.markdown("Fitur mana ('Sales' atau 'Profit') yang memiliki hubungan statistik terkuat dengan 'Quantity'?")
    
    feature_scores_df = run_feature_selection(df_clean)
    
    if not feature_scores_df.empty:
        st.dataframe(feature_scores_df, use_container_width=True)
        st.caption("Metode: `SelectKBest` dengan `f_regression`. F-Score yang lebih tinggi menunjukkan fitur yang lebih baik untuk prediksi.")
        # Tentukan fitur terbaik secara default
        default_feature = feature_scores_df.iloc[0]['Fitur']
    else:
        st.warning("Gagal menjalankan analisis fitur.")
        default_feature = 'Sales' # Fallback

    # --- BARU: Bagian 2: Konfigurasi Model Interaktif ---
    st.header("2. Konfigurasi Model Interaktif")
    st.markdown("Pilih fitur yang akan digunakan sebagai Sumbu X (prediktor) untuk model SVR.")
    
    feature_list = ['Sales', 'Profit']
    feature_labels = {
        'Sales': 'Penjualan (Sales)',
        'Profit': 'Keuntungan (Profit)'
    }
    
    # Set index default ke fitur terbaik dari analisis
    default_index = feature_list.index(default_feature)
    selected_feature = st.selectbox(
        "Pilih Fitur (Sumbu X) untuk Regresi:",
        feature_list,
        index=default_index,
        format_func=lambda x: feature_labels[x] # Tampilkan label yang mudah dibaca
    )
    selected_label = feature_labels[selected_feature]

    # --- Bagian 3: Menampilkan Metrik Evaluasi ---
    st.header(f"3. Perbandingan Metrik (Prediksi Quantity berdasarkan {selected_label})")
    
    # PERBAIKAN: Kirim 'selected_feature' ke fungsi analisis
    df_metrics, df_plot_final = run_all_svr_analysis(df_clean, selected_feature)
    
    if not df_metrics.empty:
        df_metrics_display = df_metrics.copy()
        df_metrics_display['MSE'] = df_metrics_display['MSE'].round(2)
        df_metrics_display['R2'] = df_metrics_display['R2'].round(4)
        df_metrics_display['MAPE'] = df_metrics_display['MAPE'].round(2).astype(str) + ' %'
        st.dataframe(df_metrics_display.sort_values(by='R2', ascending=False), use_container_width=True)
    else:
        st.warning("Gagal menghitung metrik model.")

    # --- Bagian 4: Visualisasi Hyperplane (Semua Kernel Default) ---
    st.header("4. Visualisasi Hyperplane Regresi (Default Parameters)")
    if not df_plot_final.empty:
        col1, col2 = st.columns(2)
        kernels = ['linear', 'poly', 'rbf', 'sigmoid']
        titles = ['Linear', 'Polynomial', 'RBF', 'Sigmoid']
        
        for i, (kernel, title) in enumerate(zip(kernels, titles)):
            # PERBAIKAN: Kirim 'selected_feature' dan 'selected_label' ke fungsi plot
            chart = create_svr_chart(
                df_plot_final, 
                kernel, 
                f'SVR Kernel {title}', 
                selected_feature, 
                selected_label
            )
            if chart:
                target_col = col1 if i % 2 == 0 else col2
                with target_col:
                    st.subheader(f"Kernel: {title}")
                    st.altair_chart(chart, use_container_width=True)

    # --- Bagian 5: Interaktivitas: Custom Kernel ---
    st.header("5. Interaktif: Uji Coba Kernel dan Tuning Parameter")
    st.sidebar.header("Tuning Parameter SVR")
    
    selected_kernel = st.sidebar.selectbox("Pilih Kernel", ['rbf', 'linear', 'poly', 'sigmoid'], index=0)
    c_value = st.sidebar.slider("Pilih nilai C (Regularisasi)", 0.1, 100.0, 50.0)
    
    gamma_value = 'auto'
    if selected_kernel in ['rbf', 'poly', 'sigmoid']:
        gamma_value = st.sidebar.slider("Pilih nilai Gamma", 0.001, 10.0, 0.1)

    degree_value = 3
    coef0_value = 0.0
    if selected_kernel == 'poly':
        degree_value = st.sidebar.slider("Pilih Degree (untuk Poly)", 1, 5, 2)
    if selected_kernel == 'sigmoid':
        coef0_value = st.sidebar.slider("Pilih Coef0 (untuk Sigmoid)", -10.0, 10.0, 0.0)

    # PERBAIKAN: Kirim 'selected_feature' ke fungsi kustom
    df_plot_custom = run_custom_svr(
        df_clean, selected_feature, selected_kernel, c_value, 
        gamma_value, degree_value, coef0_value
    )
    
    custom_chart = create_svr_chart(
        df_plot_custom, 
        selected_kernel, 
        f'SVR Kustom: Kernel {selected_kernel.capitalize()}, C={c_value:.1f}', 
        selected_feature, 
        selected_label
    )
    if custom_chart: st.altair_chart(custom_chart, use_container_width=True)

    st.markdown("---")
else:
    if df_clean is not None:
        st.error(f"Data tersisa ({len(df_clean)} baris) terlalu sedikit untuk analisis SVR (minimal 10 baris diperlukan).")
