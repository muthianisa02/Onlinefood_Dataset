# app.py
import streamlit as st
import pandas as pd
import joblib
import os

# Memuat model dan preprocessor yang telah disimpan
try:
    # Menggunakan os.path.join untuk memastikan jalur file berfungsi di mana saja
    model_path = os.path.join(os.path.dirname(__file__), 'best_svc_model.pkl')
    preprocessor_path = os.path.join(os.path.dirname(__file__), 'preprocessor.pkl')
    
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    
    # Definisi label untuk hasil prediksi
    feedback_labels = {0: 'Negative', 1: 'Positive'}
    
    st.title("Aplikasi Prediksi Feedback Pelanggan")
    st.write("Aplikasi ini memprediksi apakah feedback pelanggan akan 'Positive' atau 'Negative' berdasarkan data demografi dan perilaku.")

except FileNotFoundError:
    st.error("Error: File model atau preprocessor tidak ditemukan. Pastikan file 'best_svc_model.pkl' dan 'preprocessor.pkl' ada di direktori yang sama.")
    st.stop()
    
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat model: {e}")
    st.stop()

# --- Membuat Antarmuka Pengguna (UI) di sidebar ---

st.sidebar.header("Input Data Pelanggan")

# Fitur Numerik
age = st.sidebar.number_input("Usia", min_value=18, max_value=100, value=25)
family_size = st.sidebar.number_input("Ukuran Keluarga", min_value=1, max_value=10, value=3)
latitude = st.sidebar.number_input("Latitude", value=12.977, format="%.4f")
longitude = st.sidebar.number_input("Longitude", value=77.5993, format="%.4f")
pin_code = st.sidebar.number_input("Kode Pos", min_value=100000, max_value=999999, value=560001)

# Fitur Kategorikal
gender_options = ['Female', 'Male']
gender = st.sidebar.selectbox("Jenis Kelamin", options=gender_options)

marital_status_options = ['Single', 'Married', 'Prefer not to say']
marital_status = st.sidebar.selectbox("Status Pernikahan", options=marital_status_options)

occupation_options = ['Student', 'Employee', 'Self Employed', 'House wife']
occupation = st.sidebar.selectbox("Pekerjaan", options=occupation_options)

monthly_income_options = ['No Income', 'Below Rs.10000', '10001 to 25000', '25001 to 50000', 'More than 50000']
monthly_income = st.sidebar.selectbox("Pendapatan Bulanan", options=monthly_income_options)

educational_qual_options = ['Post Graduate', 'Graduate', 'Ph.D', 'School', 'Uneducated']
educational_qual = st.sidebar.selectbox("Kualifikasi Pendidikan", options=educational_qual_options)

output_options = ['Yes', 'No']
output = st.sidebar.selectbox("Output (Apakah Pernah Memesan)", options=output_options)

# Membuat tombol prediksi
if st.sidebar.button("Prediksi Feedback"):
    # Mengumpulkan semua input ke dalam DataFrame
    input_data = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'Marital Status': marital_status,
        'Occupation': occupation,
        'Monthly Income': monthly_income,
        'Educational Qualifications': educational_qual,
        'Family size': family_size,
        'latitude': latitude,
        'longitude': longitude,
        'Pin code': pin_code,
        'Output': output
    }])

    try:
        # Menerapkan preprocessor pada data input
        processed_input = preprocessor.transform(input_data)
        
        # Melakukan prediksi
        prediction = model.predict(processed_input)
        prediction_label = feedback_labels[prediction[0]]

        # Menampilkan hasil prediksi
        st.subheader("Hasil Prediksi:")
        if prediction_label == 'Positive':
            st.success(f"Feedback yang diprediksi adalah: **{prediction_label}** 🎉")
        else:
            st.warning(f"Feedback yang diprediksi adalah: **{prediction_label}** 😞")
            
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memprediksi: {e}")