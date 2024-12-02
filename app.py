from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import pytesseract
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, jsonify

# Initialize Flask app
app = Flask(__name__)

# Define allowed file types
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Set the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Check if file type is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to extract text from an image using Tesseract
def extract_text_from_image(image_path):
    # Open the image file
    img = Image.open(image_path)
    # Use pytesseract to extract text
    extracted_text = pytesseract.image_to_string(img)
    return extracted_text

# Function to extract structured information from text (adjust this function to your needs)
def extract_structured_info(extracted_text):
    structured_info = {
        "patient_name": "Not found",
        "date": "Not found",
        "disease": "Not found",
        "test": "Not found",
        "medicine": "Not found"
    }
    
    # Simple extraction logic (You may need to enhance this)
    if "name:" in extracted_text:
        structured_info["patient_name"] = extracted_text.split("name:")[1].split("\n")[0].strip()
    
    if "date:" in extracted_text:
        structured_info["date"] = extracted_text.split("date:")[1].split("\n")[0].strip()

    # Add more extraction logic for disease, test, medicine, etc.

    return structured_info

@app.route('/')
def login():
    return render_template('login.html')
@app.route('/login', methods=['POST'])
def login_submit():
    # Here you would validate the login credentials if necessary
    # For now, we simply redirect to index.html regardless of the credentials
    return redirect(url_for('index'))
@app.route('/index')
def index():
    return render_template('index.html')  # This renders the login.html page

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    # Check if the file is allowed
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"})

    # Secure the filename and save the file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Extract text from the uploaded image
    extracted_text = extract_text_from_image(file_path)

    # Extract structured information from the text
    structured_info = extract_structured_info(extracted_text)

    # Return the output containing only structured information (no predictions)
    return jsonify({"structured_info": structured_info, "extracted_text": extracted_text})

if __name__ == '__main__':
    app.run(debug=True)
