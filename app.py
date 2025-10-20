from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import os

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
    port = int(os.environ.get("PORT", 5000))

    # ✅ Only use ngrok for local development
    if os.environ.get("RENDER") != "true":
        try:
            from pyngrok import ngrok
            ngrok.set_auth_token("34HuQjXZnGNKnmf7K0jo1FQZis2_Yq8EHrWPNxSu6yLCK7tv")
            public_url = ngrok.connect(port).public_url
            print(f"🚀 Ngrok tunnel running at: {public_url}")
        except Exception as e:
            print("⚠️ Ngrok not started:", e)

    app.run(host="0.0.0.0", port=port)
