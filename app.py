import os
import streamlit as st

# --- 1. CONFIG PEMANGGILAN PERTAMA ---
st.set_page_config(page_title="Herbal Scan AI", page_icon="🌿", layout="wide")

# --- 2. SET PATH CACHE ---
os.environ['TORCH_HOME'] = 'F:/Porto/indo_medicinal_plant/torch_cache'

import torch
from torchvision import models, transforms
from PIL import Image
import joblib
import numpy as np
import pandas as pd

# --- 3. LOAD ASSETS ---
@st.cache_resource
def load_assets():
    model_ml = joblib.load('hasil model/best_model_SVM.pkl') 
    scaler = joblib.load('hasil model/scaler_model_best.pkl')
    le = joblib.load('hasil model/label_encoder_best.pkl')
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    extractor = efficientnet.features.to(device).eval()
    pooler = torch.nn.AdaptiveAvgPool2d(1)
    
    return model_ml, scaler, le, extractor, pooler, device

try:
    model_ml, scaler, le, extractor, pooler, device = load_assets()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# --- 4. SIDEBAR (LOG PROSES) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/628/628283.png", width=100)
    st.title("Detail Proyek")
    st.info("""
    **Metodologi yang Digunakan:**
    0. **Dataset**: 10.000+ gambar tanaman herbal Indonesia, 20 kelas. Diambil dari "https://github.com/Salmanim20/indo_medicinal_plant"
    1. **Preprocessing**: Resize 224x224 & Image Normalization.
    2. **Feature Extraction**: Menggunakan **EfficientNet-B0** (Transfer Learning) untuk mengambil fitur esensial gambar.
    3. **Dimensionality**: Menghasilkan 1280 fitur per gambar.
    4. **Classification**: **SVM (Support Vector Machine)** sebagai classifier akhir.
    """)
    
    st.success(f"Running on: **{device.upper()}**")
    st.write("---")
    st.caption("Developed by: Affandi (Data Science Student)")

# --- 5. UI UTAMA ---
st.title("🌿 Herbal Scan: Identifikasi Tanaman Obat")
st.markdown("Aplikasi ini menggunakan **Hybrid Deep Learning** untuk mengenali jenis tanaman herbal Indonesia melalui foto.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📸 Input Gambar")
    uploaded_file = st.file_uploader("Pilih foto daun/tanaman...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption='Gambar Terunggah', use_container_width=True)

with col2:
    st.subheader("🔍 Hasil Analisis AI")
    if uploaded_file is not None:
        if st.button('Mulai Klasifikasi'):
            with st.spinner('Proses ekstraksi fitur sedang berjalan...'):
                # Preprocessing
                preprocess = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                
                img_t = preprocess(img).unsqueeze(0).to(device)
                
                # Ekstraksi Fitur
                with torch.no_grad():
                    feat = extractor(img_t)
                    feat = pooler(feat).view(1, -1)
                    feat_np = feat.cpu().numpy()

                # Scaling & Prediksi
                feature_names = [f"f_{i}" for i in range(feat_np.shape[1])]
                feat_df = pd.DataFrame(feat_np, columns=feature_names)
                feat_scaled = scaler.transform(feat_df)
                
                pred_idx = model_ml.predict(feat_scaled)
                nama_tanaman = le.inverse_transform(pred_idx)[0]

                # Output Visual
                st.success(f"### Tanaman Terdeteksi: **{nama_tanaman}**")
                
                # Tampilkan Keyakinan (Probabilitas)
                try:
                    prob = model_ml.predict_proba(feat_scaled)[0]
                    top_idx = np.argmax(prob)
                    confidence = prob[top_idx] * 100
                    
                    st.metric(label="Tingkat Keyakinan", value=f"{confidence:.2f}%")
                    st.progress(confidence / 100)
                except:
                    pass
                
                st.write("---")
                st.markdown(f"""
                **Apa selanjutnya?**
                - Tanaman **{nama_tanaman}** dikenal dalam pengobatan tradisional.
                - Gunakan hasil ini sebagai referensi awal, tetap konsultasikan dengan ahli botani/medis.
                """)
    else:
        st.info("💡 Silakan upload gambar tanaman di panel sebelah kiri untuk memulai.")

# --- 6. FOOTER ---
st.divider()
st.center = st.write("© 2026 | Build with Streamlit & PyTorch")