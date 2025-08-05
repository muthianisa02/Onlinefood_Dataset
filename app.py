# app.py
import flask
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Memuat model dan preprocessor yang telah disimpan
# Pastikan file-file ini ada di direktori yang sama dengan app.py saat deployment
try:
    model = joblib.load('best_svc_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    # LabelEncoder juga perlu dimuat atau dibuat ulang untuk mapping
    # Asumsi label_encoder dari pelatihan: 0 untuk 'Negative', 1 untuk 'Positive'
    # Jika Anda menyimpan label_encoder, muat di sini. Jika tidak, buat manual.
    # Untuk kasus ini, kita tahu kelasnya biner, jadi bisa manual.
    feedback_labels = {0: 'Negative', 1: 'Positive'}
    print("Model dan preprocessor berhasil dimuat.")
except Exception as e:
    print(f"Error saat memuat model atau preprocessor: {e}")
    model = None
    preprocessor = None
    feedback_labels = {0: 'Negative', 1: 'Positive'} # Default jika gagal muat

@app.route('/')
def home():
    """Halaman utama untuk antarmuka sederhana."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint untuk melakukan prediksi."""
    if model is None or preprocessor is None:
        return jsonify({'error': 'Model atau preprocessor belum dimuat. Periksa log server.'}), 500

    try:
        data = request.json
        
        # Contoh data yang diharapkan:
        # {
        #     "Age": 25,
        #     "Gender": "Female",
        #     "Marital Status": "Single",
        #     "Occupation": "Student",
        #     "Monthly Income": "No Income",
        #     "Educational Qualifications": "Graduate",
        #     "Family size": 3,
        #     "latitude": 12.977,
        #     "longitude": 77.5773,
        #     "Pin code": 560009,
        #     "Output": "Yes"
        # }

        # Buat DataFrame dari input JSON
        input_df = pd.DataFrame([data])

        # Pastikan kolom sesuai urutan dan nama yang diharapkan oleh preprocessor
        # (Ini penting karena OneHotEncoder akan membuat kolom berdasarkan urutan saat fit)
        # Ambil kolom asli dari X_train yang digunakan saat fit preprocessor
        # Untuk ini, kita perlu mengetahui kolom asli dari X
        # Jika Anda tidak menyimpan X_train, Anda harus memastikan input_df memiliki kolom yang sama
        # dengan urutan yang sama seperti df.drop('Feedback', axis=1)
        
        # Contoh kolom yang diharapkan (sesuai dengan X dari stage 2)
        expected_columns = [
            'Age', 'Gender', 'Marital Status', 'Occupation', 'Monthly Income',
            'Educational Qualifications', 'Family size', 'latitude', 'longitude',
            'Pin code', 'Output'
        ]
        
        # Pastikan input_df memiliki semua kolom yang diharapkan, isi NaN jika tidak ada
        # dan urutkan sesuai expected_columns
        input_df = input_df.reindex(columns=expected_columns, fill_value=np.nan)

        # Lakukan preprocessing pada data input
        # Gunakan preprocessor.transform, BUKAN fit_transform
        processed_input = preprocessor.transform(input_df)

        # Lakukan prediksi
        prediction_encoded = model.predict(processed_input)[0]
        prediction_label = feedback_labels.get(prediction_encoded, "Unknown")

        return jsonify({'prediction': prediction_label})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Untuk pengembangan lokal, gunakan:
    # app.run(debug=True, host='0.0.0.0', port=5000)
    # Untuk deployment di Heroku, Gunicorn akan menjalankan app
    app.run(debug=False) # Pastikan debug=False saat deployment