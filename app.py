
from pyngrok import ngrok

ngrok.set_auth_token("34HuQjXZnGNKnmf7K0jo1FQZis2_Yq8EHrWPNxSu6yLCK7tv")


from flask import Flask, request, jsonify
from flask_cors import CORS  # ✅ Import CORS
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from pyngrok import ngrok

app = Flask(__name__)
CORS(app)  # ✅ Enable CORS for all routes

# Load model and encoders
model = tf.keras.models.load_model("upi_fraud_cnn.h5")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        df = pd.DataFrame([data])

        # 🛠 Safe label encoding helper
        def safe_transform(value, encoder):
            if value not in encoder.classes_:
                encoder.classes_ = np.append(encoder.classes_, value)
            return encoder.transform([value])[0]

        # ✅ Encode categorical columns safely
        df['category'] = df['category'].apply(lambda x: safe_transform(x, le))
        df['state'] = df['state'].apply(lambda x: safe_transform(x, le))

        # 🧮 Scale and reshape
        X_scaled = scaler.transform(df)
        X_cnn = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

        # 🔮 Predict
        prob = model.predict(X_cnn)[0][0]
        prediction = int(prob > 0.5)

        return jsonify({
            'fraud_flag': prediction,
            'probability': float(prob)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    public_url = ngrok.connect(5000)
    print(f"🚀 Public URL: {public_url}")
    app.run(port=5000)
