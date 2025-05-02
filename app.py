from flask import Flask, render_template, request, jsonify
import os
import torch
from model.model import load_model, predict_sequence
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.secret_key = 'your_secret_key_here'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model('model/best_vit_sequence_model.pth', device)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'sequence' not in request.files:
        return jsonify({'error': 'No sequence uploaded'}), 400
    
    sequence = request.files.getlist('sequence')
    file_paths = []

    # Save uploaded files
    for file in sequence:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        file_paths.append(file_path)

    # Make predictions
    try:
        prediction = predict_sequence(model, file_paths, device)
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)