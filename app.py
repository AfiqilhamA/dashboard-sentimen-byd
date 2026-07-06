import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Dashboard Analisis Sentimen", layout="wide")

# --- 2. CSS CUSTOM (TEMA MODERN DARK - ENTERPRISE) ---
st.markdown("""
    <style>
    .stApp { background-color: #0F172A !important; }
    h1, h2, h3, h4, p, label, .stMarkdown, .stTableCell { color: #F8FAFC !important; font-family: 'Segoe UI', sans-serif; }
    
    [data-testid="stMetric"] {
        background-color: #1E293B !important;
        padding: 20px !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3) !important;
        border-left: 4px solid #38BDF8 !important;
    }
    [data-testid="stMetricValue"] { color: #F8FAFC !important; }

    .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: #0F172A; }
    .stTabs [data-baseweb="tab"] { background-color: #1E293B !important; color: #94A3B8 !important; border-radius: 6px 6px 0px 0px !important; border: none !important; }
    .stTabs [aria-selected="true"] { background-color: #38BDF8 !important; color: #0F172A !important; font-weight: bold; }
    .stTextArea textarea, .stTextInput input { background-color: #1E293B !important; color: #F8FAFC !important; border: 1px solid #334155 !important; border-radius: 6px !important; }
    
    .viral-card {
        background-color: #1E293B;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        border-left: 4px solid;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. INIT SESSION STATE ---
if 'history' not in st.session_state:
    st.session_state.history = []

# --- 4. LOAD MODEL ---
@st.cache_resource
def load_model():
    # Pastikan load file ini karena file ini yang menyimpan Grid Search CV terbaik
    with open('sentiment_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

try:
    model_ai = load_model()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")

# --- 5. LOAD DATA (DISAMAKAN 100% DENGAN JUPYTER) ---
@st.cache_data
def load_data():
    df_raw = pd.read_csv("rawdata.csv")
    df_labelled = pd.read_csv("labelledYT (5).csv") 
    
    if 'Timestamp' in df_labelled.columns:
        df_labelled['Timestamp'] = pd.to_datetime(df_labelled['Timestamp'], errors='coerce')
    
    if 'Likes' in df_labelled.columns:
        df_labelled['Likes'] = pd.to_numeric(df_labelled['Likes'], errors='coerce').fillna(0)
    if 'score' in df_labelled.columns:
        df_labelled['score'] = pd.to_numeric(df_labelled['score'], errors='coerce').fillna(0)
        
    # Murni menggunakan kolom 'label' asli CSV tanpa menghitung skor lagi
    df_labelled['label_clean'] = df_labelled['label'].astype(str).str.lower().str.strip()
        
    return df_raw, df_labelled

try:
    df_raw, df_labelled = load_data()
    counts = df_labelled['label_clean'].value_counts()
    
    positif_count = counts.get('positif', 0)
    negatif_count = counts.get('negatif', 0)
    netral_count = counts.get('netral', 0)
    
    total_valid = positif_count + negatif_count + netral_count
    if total_valid == 0: total_valid = 1 
    
    pos_pct = (positif_count / total_valid) * 100
    neg_pct = (negatif_count / total_valid) * 100
    net_pct = (netral_count / total_valid) * 100
except Exception as e:
    st.error("Gagal memuat data.")
    total_valid, positif_count, negatif_count, netral_count = 0, 0, 0, 0
    pos_pct, neg_pct, net_pct = 0, 0, 0

# --- 6. MENGHITUNG AKURASI MODEL (REPLIKASI DINAMIS DARI JUPYTER) ---
try:
    # 1. Drop NA persis kayak di Jupyter
    df_eval = df_labelled.dropna(subset=['FinalNormalizer', 'label_clean']).copy()
    
    # 2. Ambil data positif dan negatif
    df_pos = df_eval[(df_eval["label_clean"] == "positif") | (df_eval["label_clean"] == "positive")]
    df_neg = df_eval[(df_eval["label_clean"] == "negatif") | (df_eval["label_clean"] == "negative")]
    
    # 3. Lakukan Mean Resampling & Balancing data
    target_n = int(abs((len(df_pos) + len(df_neg)) / 2))
    df_pos_bal = resample(df_pos, replace=True, n_samples=target_n, random_state=42)
    df_neg_bal = resample(df_neg, replace=True, n_samples=target_n, random_state=42)
    df_balanced = pd.concat([df_pos_bal, df_neg_bal]).sample(frac=1, random_state=42)
    
    # 4. Train-Test Split (Gunakan 30% Test Size untuk diuji)
    X = df_balanced["FinalNormalizer"].astype(str)
    y = df_balanced["label_clean"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 5. Prediksi murni menggunakan Test Set
    y_true = y_test.tolist()
    preds = model_ai.predict(X_test.tolist())
    y_pred = [str(p).lower().strip() for p in preds]
    
    # 6. Hitung skor akurasi
    correct_predictions = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    acc_score = (correct_predictions / len(y_true)) * 100 if len(y_true) > 0 else 0
except Exception as e:
    acc_score = 0.0

# --- 7. HEADER DASHBOARD ---
st.markdown("<h1>Dashboard Analisis Sentimen</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #94A3B8; font-size: 16px;'>Monitoring pergerakan opini publik dan evaluasi performa Machine Learning.</p>", unsafe_allow_html=True)
st.write("") 

# --- 8. METRIK ANGKA ---
st.markdown("### Ringkasan Performa & Data")

col_top1, col_top2 = st.columns(2)
with col_top1:
    st.metric(label="Total Data Terekam", value=f"{total_valid:,} Baris")
with col_top2:
    st.metric(label="🎯 Akurasi Model", value=f"{acc_score:.2f}%", delta="Tingkat Keakuratan Prediksi")

st.write("")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Sentimen Positif", value=f"{positif_count:,}", delta=f"{pos_pct:.1f}%")
with col2:
    st.metric(label="Sentimen Netral", value=f"{netral_count:,}", delta=f"{net_pct:.1f}%", delta_color="off")
with col3:
    st.metric(label="Sentimen Negatif", value=f"{negatif_count:,}", delta=f"-{neg_pct:.1f}%", delta_color="inverse")

st.markdown("<br>", unsafe_allow_html=True)

# --- 9. VISUALISASI CHART UTAMA ---
col_gauge, col_donut, col_bar = st.columns(3)

color_pos, color_net, color_neg = "#10B981", "#FBBF24", "#F43F5E"
color_map = {'Positif': color_pos, 'Netral': color_net, 'Negatif': color_neg}

df_chart = pd.DataFrame({
    'Sentimen': ['Positif', 'Netral', 'Negatif'],
    'Jumlah': [positif_count, netral_count, negatif_count]
})
df_chart = df_chart[df_chart['Jumlah'] > 0]

with col_gauge:
    st.markdown("#### Skor Dominasi Positif")
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number", value = pos_pct,
        number = {'suffix': "%", 'font': {'size': 50, 'color': color_pos}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': color_pos}, 'bgcolor': "#1E293B",
            'borderwidth': 2, 'bordercolor': "#334155",
            'steps': [{'range': [0, 50], 'color': '#334155'}, {'range': [50, 100], 'color': '#0F172A'}],
        }
    ))
    fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "#F8FAFC"}, height=300, margin=dict(t=20, b=10, l=10, r=10))
    st.plotly_chart(fig_gauge, use_container_width=True)

with col_donut:
    st.markdown("#### Proporsi & Jumlah")
    fig_donut = px.pie(df_chart, names='Sentimen', values='Jumlah', hole=0.55, color='Sentimen', color_discrete_map=color_map)
    fig_donut.update_traces(textposition='inside', textinfo='label+value+percent', textfont_size=13)
    fig_donut.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': '#F8FAFC'}, showlegend=False, height=300, margin=dict(t=20, b=10, l=10, r=10))
    st.plotly_chart(fig_donut, use_container_width=True)

with col_bar:
    st.markdown("#### Perbandingan Bar")
    fig_bar = px.bar(df_chart, x='Sentimen', y='Jumlah', text='Jumlah', color='Sentimen', color_discrete_map=color_map)
    fig_bar.update_traces(textposition='outside', textfont_size=14)
    fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': '#F8FAFC'}, showlegend=False, height=300, margin=dict(t=20, b=10, l=10, r=10))
    fig_bar.update_xaxes(showgrid=False, title="")
    fig_bar.update_yaxes(showgrid=True, gridcolor='#334155', title="")
    st.plotly_chart(fig_bar, use_container_width=True)

st.write("---")

# --- 10. FITUR WORD CLOUD ---
st.markdown("### ☁️ Kata Paling Sering Muncul (Word Cloud)")
col_wc1, col_wc2 = st.columns(2)

def plot_wordcloud(text, colormap):
    if str(text).strip() == "" or str(text) == "nan":
        return None
    wc = WordCloud(width=500, height=350, background_color='#1E293B', colormap=colormap, max_words=100, collocations=False).generate(text)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    fig.patch.set_facecolor('#1E293B') 
    return fig

with col_wc1:
    st.markdown(f"<h4 style='color:{color_pos}; text-align:center;'>🟢 Topik Sentimen Positif</h4>", unsafe_allow_html=True)
    text_pos = " ".join(df_labelled[df_labelled['label_clean'] == 'positif']['FinalNormalizer'].dropna().astype(str))
    fig_pos = plot_wordcloud(text_pos, 'Greens')
    if fig_pos:
        st.pyplot(fig_pos)
    else:
        st.info("Data kata positif kosong.")

with col_wc2:
    st.markdown(f"<h4 style='color:{color_neg}; text-align:center;'>🔴 Topik Sentimen Negatif</h4>", unsafe_allow_html=True)
    text_neg = " ".join(df_labelled[df_labelled['label_clean'] == 'negatif']['FinalNormalizer'].dropna().astype(str))
    fig_neg = plot_wordcloud(text_neg, 'Reds')
    if fig_neg:
        st.pyplot(fig_neg)
    else:
        st.info("Data kata negatif kosong.")

st.write("---")

# --- 11. GRAFIK TREN WAKTU ---
if 'Timestamp' in df_labelled.columns:
    st.markdown("### 📈 Tren Sentimen Seiring Waktu")
    
    df_time = df_labelled.dropna(subset=['Timestamp']).copy()
    df_time['Date'] = df_time['Timestamp'].dt.date
    trend_data = df_time.groupby(['Date', 'label_clean']).size().reset_index(name='Jumlah')
    trend_data['Sentimen'] = trend_data['label_clean'].str.title()
    
    fig_line = px.line(trend_data, x='Date', y='Jumlah', color='Sentimen', 
                       color_discrete_map=color_map, markers=True)
    fig_line.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                           font={'color': '#F8FAFC'}, height=350, margin=dict(t=10, b=10, l=10, r=10))
    fig_line.update_xaxes(showgrid=False, title="Tanggal")
    fig_line.update_yaxes(showgrid=True, gridcolor='#334155', title="Jumlah Komentar")
    
    st.plotly_chart(fig_line, use_container_width=True)
    st.write("---")

# --- 12. CONTOH KOMENTAR ---
if 'score' in df_labelled.columns and 'label_clean' in df_labelled.columns:
    st.markdown("### 🔥 Contoh Komentar Positif dan Negatif")
    
    # 1. Copy data dan pastikan format teks
    df_contoh = df_labelled.dropna(subset=['Comment', 'score']).copy()
    df_contoh['Comment'] = df_contoh['Comment'].astype(str)
    
    # 2. FILTER PINTAR: Buang komentar super pendek (<15 huruf) dan cerpen (>250 huruf)
    df_contoh = df_contoh[(df_contoh['Comment'].str.len() >= 15) & (df_contoh['Comment'].str.len() <= 250)]
    
    # 3. Pisahkan Positif dan Negatif
    pos_pool = df_contoh[df_contoh['label_clean'].isin(['positif', 'positive'])].copy()
    neg_pool = df_contoh[df_contoh['label_clean'].isin(['negatif', 'negative'])].copy()
    
    # 4. FILTER ANTI-DRAMA & ANTI-OOT (Khusus Negatif)
    # Buang komentar toxic, kompetitor, debat propaganda, dan teori konspirasi
    exclude_words = [
        '@', 'fanboy', 'bacot', 'buzzer', 'tolol', 'bego', 'goblok', 'siapa', 
        'zenix', 'innova', 'wuling', 'ioniq', 'air ev', 'sales',
        'propaganda', 'amrik', 'molis cina', 'cina', 'china', 'politik',
        'konspirasi', 'bbm', 'hanif', 'disetting', 'berita'
    ]
    for word in exclude_words:
        neg_pool = neg_pool[~neg_pool['Comment'].str.contains(word, case=False, na=False, regex=False)]
        
    # Tambahan filter khusus Regex biar kata 'ice' gak ngehapus 'service' dan 'bang' gak ngehapus 'bangku'
    regex_words = [r'\bice\b', r'\bbang\b']
    for word in regex_words:
        neg_pool = neg_pool[~neg_pool['Comment'].str.contains(word, case=False, na=False, regex=True)]
    
    # 5. AMBIL YANG PALING TEGAS (Skor tertinggi & terendah, BUKAN random)
    top_pos = pos_pool.nlargest(3, 'score') if not pos_pool.empty else pos_pool
    top_neg = neg_pool.nsmallest(3, 'score') if not neg_pool.empty else neg_pool
    
    col_viral1, col_viral2 = st.columns(2, gap="large")
    
    with col_viral1:
        st.markdown(f"<h4 style='color:{color_pos};'>🟢 Komentar Positif</h4>", unsafe_allow_html=True)
        for _, row in top_pos.iterrows():
            likes_text = f"👍 {int(row['Likes'])} Likes | " if 'Likes' in row and pd.notna(row['Likes']) else ""
            st.markdown(f"""
            <div class='viral-card' style='border-color:{color_pos};'>
                <p style='font-style:italic; font-size:14px; margin-bottom:5px;'>"{row['Comment']}"</p>
                <b style='color:#38BDF8;'>{likes_text} 🎯 Skor: {row['score']}</b>
            </div>
            """, unsafe_allow_html=True)
            
    with col_viral2:
        st.markdown(f"<h4 style='color:{color_neg};'>🔴 Komentar Negatif</h4>", unsafe_allow_html=True)
        for _, row in top_neg.iterrows():
            likes_text = f"👍 {int(row['Likes'])} Likes | " if 'Likes' in row and pd.notna(row['Likes']) else ""
            st.markdown(f"""
            <div class='viral-card' style='border-color:{color_neg};'>
                <p style='font-style:italic; font-size:14px; margin-bottom:5px;'>"{row['Comment']}"</p>
                <b style='color:#38BDF8;'>{likes_text} 🎯 Skor: {row['score']}</b>
            </div>
            """, unsafe_allow_html=True)
            
    st.write("---")
# --- 13. ETALASE DATA & DOWNLOAD ---
st.markdown("### 🗂️ Database Log")
tab1, tab2, tab3 = st.tabs(["Raw Data", "Preprocessed", "Label & Score"])

with tab1:
    st.dataframe(df_raw.head(100), use_container_width=True)
with tab2:
    st.dataframe(df_labelled[['Comment', 'PreposText', 'stemmed', 'FinalNormalizer']].head(100), use_container_width=True)
with tab3:
    st.dataframe(df_labelled.head(100), use_container_width=True)
    csv = df_labelled.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download Data Prediksi (CSV)", data=csv, file_name='data_sentimen.csv', mime='text/csv')

st.markdown("---")

# --- 14. UJI COBA MODEL (HISTORY) ---
st.markdown("### 💬 Live Pengujian Model")
st.write("Sistem akan merekam setiap teks yang Anda uji dalam sesi ini.")

col_input, col_history = st.columns([1, 1], gap="large")

with col_input:
    user_input = st.text_area("Input teks pengujian model:", placeholder="Ketik komentar di sini untuk dianalisis...", height=120)

    if st.button("Jalankan Analisis", use_container_width=True):
        if user_input:
            hasil = model_ai.predict([user_input])
            sentimen = str(hasil[0]).title()
            
            if sentimen == 'Positif':
                st.success(f"Status Prediksi: {sentimen}")
            elif sentimen == 'Netral':
                st.info(f"Status Prediksi: {sentimen}")
            else:
                st.error(f"Status Prediksi: {sentimen}")
            
            st.session_state.history.append({"Teks Komentar": user_input, "Hasil Sentimen": sentimen})
        else:
            st.warning("Kolom teks masih kosong, silakan isi terlebih dahulu!")

with col_history:
    st.markdown("#### Tabel Riwayat Uji Coba")
    if st.session_state.history:
        df_history = pd.DataFrame(st.session_state.history)
        st.dataframe(df_history, use_container_width=True, height=200)
        
        if st.button("Hapus Riwayat"):
            st.session_state.history = []
            st.rerun() 
    else:
        st.info("Belum ada data pengujian. Silakan tes model pada kolom di samping.")
