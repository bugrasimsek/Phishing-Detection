from flask import Flask, render_template, request
import xgboost as xgb
import joblib
import os
import trafilatura
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Load your XGBoost model
model = xgb.XGBClassifier()
model.load_model('model/xlm_roberta_xgb_model.json')  # Use XGBoost's load_model function

# Load Sentence Transformer model
st_model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'htmlFile' not in request.files:
        return "No file part"

    # Here file is a FileStorage object which has the 'filename' attribute.
    file = request.files['htmlFile']
    if file.filename == '':
        return "No selected file"

    # Save the uploaded file to a temporary location
    file_path = os.path.join('test', file.filename)
    file.save(file_path)

    # Extract text from HTML
    with open(file_path, 'r', encoding='utf-8') as html_file:
        html_content = html_file.read()

    text = trafilatura.extract(html_content)

    # Ensure text is extracted properly
    if not text:
        return "Unable to extract text from HTML"

    # Generate embedding
    embedding = st_model.encode(text, convert_to_tensor=True)
    embedding = embedding.cpu().numpy().reshape(1, -1)

    # Predict using the trained model
    prediction = model.predict(embedding)

    if prediction[0] == 1:
        prediction_result = "Phishing"
    else:
        prediction_result = "Legitimate"


    # Return the filename and the prediction result
    return f"{file.filename} is {prediction_result}"


if __name__ == '__main__':
    app.run(debug=True, port=5050)
