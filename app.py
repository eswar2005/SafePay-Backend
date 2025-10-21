# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import numpy as np
# import pandas as pd
# import joblib
# import tensorflow as tf
# import os

# app = Flask(__name__)
# CORS(app)  # ✅ Enable CORS for all routes

# # Load model and encoders
# model = tf.keras.models.load_model("upi_fraud_cnn.h5")
# scaler = joblib.load("scaler.pkl")
# le = joblib.load("label_encoder.pkl")

# @app.route('/')
# def home():
#     return "Flask API is running!"
# @app.route('/predict', methods=['POST','GET'])
# def predict():
#     if request.method == 'GET':
#         return jsonify({"message": "Send a POST request with transaction data."})
#     try:
#         data = request.get_json(force=True)
#         df = pd.DataFrame([data])

#         # 🛠 Safe label encoding helper
#         def safe_transform(value, encoder):
#             if value not in encoder.classes_:
#                 encoder.classes_ = np.append(encoder.classes_, value)
#             return encoder.transform([value])[0]

#         # ✅ Encode categorical columns safely
#         df['category'] = df['category'].apply(lambda x: safe_transform(x, le))
#         df['state'] = df['state'].apply(lambda x: safe_transform(x, le))

#         # 🧮 Scale and reshape
#         X_scaled = scaler.transform(df)
#         X_cnn = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

#         # 🔮 Predict
#         prob = model.predict(X_cnn)[0][0]
#         prediction = int(prob > 0.5)

#         return jsonify({
#             'fraud_flag': prediction,
#             'probability': float(prob)
#         })

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500


# if __name__ == "__main__":
#     import os
#     port = int(os.environ.get("PORT", 5000))
#     app.run(host="0.0.0.0", port=port)




from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import os

app = Flask(__name__)
CORS(app)  # ✅ Enable CORS for all routes

print("Starting Flask app...")

# Load model and encoders with logging
try:
    print("Loading TensorFlow model...")
    model = tf.keras.models.load_model("upi_fraud_cnn.h5")
    print("✅ Model loaded successfully!")

    print("Loading scaler...")
    scaler = joblib.load("scaler.pkl")
    print("✅ Scaler loaded successfully!")

    print("Loading label encoder...")
    le = joblib.load("label_encoder.pkl")
    print("✅ Label encoder loaded successfully!")

except Exception as e:
    print("❌ Error loading model or encoders:", e)
    raise e  # Crash immediately so Render logs it

@app.route('/')
def home():
    print("Home route accessed")
    return "Flask API is running!"

@app.route('/predict', methods=['POST','GET'])
def predict():
    print(f"Request method: {request.method}")

    if request.method == 'GET':
        print("GET request received at /predict")
        return jsonify({"message": "Send a POST request with transaction data."})

    try:
        print("Parsing JSON data from request...")
        data = request.get_json(force=True)
        print("Received data:", data)

        df = pd.DataFrame([data])
        print("Data converted to DataFrame:\n", df)

        # 🛠 Safe label encoding helper
        def safe_transform(value, encoder):
            if value not in encoder.classes_:
                print(f"Adding unknown category '{value}' to encoder classes")
                encoder.classes_ = np.append(encoder.classes_, value)
            return encoder.transform([value])[0]

        # Encode categorical columns safely
        print("Encoding 'category' column...")
        df['category'] = df['category'].apply(lambda x: safe_transform(x, le))
        print("Encoded 'category':", df['category'].tolist())

        print("Encoding 'state' column...")
        df['state'] = df['state'].apply(lambda x: safe_transform(x, le))
        print("Encoded 'state':", df['state'].tolist())

        # Scale and reshape
        print("Scaling data...")
        X_scaled = scaler.transform(df)
        print("Scaled data:\n", X_scaled)

        X_cnn = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
        print("Reshaped data for CNN input:", X_cnn.shape)

        # Predict
        print("Making prediction...")
        prob = model.predict(X_cnn)[0][0]
        prediction = int(prob > 0.5)
        print(f"Prediction: {prediction}, Probability: {prob}")

        return jsonify({
            'fraud_flag': prediction,
            'probability': float(prob)
        })

    except Exception as e:
        print("❌ Error during prediction:", e)
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Running Flask app on 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)

