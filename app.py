# ============================================================
# app.py - PlantCare AI Flask Backend
# ============================================================

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import tensorflow as tf
import json
import os
import secrets
import google.generativeai as genai
import requests

# ============================================================
# INITIALIZE FLASK APP
# ============================================================
app = Flask(__name__)

# Secret key for session management
app.secret_key = secrets.token_hex(16)

# ============================================================
# CONFIGURATION
# ============================================================
# Folder where uploaded images are temporarily stored
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')

# Folder where images are stored for display
STATIC_FOLDER = os.path.join(os.path.dirname(__file__), 'static')

# Allowed image formats
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Image size MobileNetV2 expects
IMG_SIZE = (224, 224)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max 16MB upload

GEMINI_API_KEY = "AIzaSyCuzbaOP3JTIAkpChKzrWsEUY8ctO3GtG8"
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.5-flash')

OPENWEATHER_API_KEY = "ac126cf002ca55b7e321f762d6e9014e"

# ============================================================
# LOAD MODEL AND CLASS LABELS
# ============================================================
model = None
class_labels = None
class_indices = None

def load_model():
    """Load the trained model and class labels"""
    global model, class_labels, class_indices

    print("Loading PlantCare AI model...")

    # Load trained model
    model_path = os.path.join(os.path.dirname(__file__), 'model/mobilenetv2_best.keras')
    model = tf.keras.models.load_model(model_path)
    print(f"   Model loaded from: {model_path}")

    # Load class indices
    labels_path = os.path.join(os.path.dirname(__file__), 'model/class_indices.json')
    with open(labels_path, 'r') as f:
        class_indices = json.load(f)

    # Reverse: index -> class name
    class_labels = {v: k for k, v in class_indices.items()}
    print(f"   Classes loaded: {len(class_labels)} disease classes")
    print("   Model ready!")

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def allowed_file(filename):
    """Check if uploaded file is an allowed image format"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path):
    """
    Preprocess image for MobileNetV2 prediction
    - Load image
    - Resize to 224x224
    - Convert to numpy array
    - Apply MobileNetV2 preprocessing
    """
    # Load and resize image
    img = Image.open(image_path).convert('RGB')
    img = img.resize(IMG_SIZE)

    # Convert to numpy array
    img_array = np.array(img, dtype=np.float32)

    # Add batch dimension (model expects 4D input)
    img_array = np.expand_dims(img_array, axis=0)

    # Apply MobileNetV2 preprocessing (scales to -1 to 1)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    return img_array


def predict_disease(image_path):
    """
    Run model prediction on uploaded image
    Returns dict with plant, disease, confidence, recommendations
    """
    # Preprocess the image
    img_array = preprocess_image(image_path)

    # Run prediction
    predictions = model.predict(img_array, verbose=0)

    # Get top prediction
    predicted_idx = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_idx]) * 100

    # Get class name
    raw_class = class_labels[predicted_idx]

    # Parse plant and disease from class name
    # Format is: PlantName___DiseaseName
    if '___' in raw_class:
        parts = raw_class.split('___')
        plant_name = parts[0].replace('_', ' ')
        disease_name = parts[1].replace('_', ' ')
    else:
        plant_name = 'Unknown'
        disease_name = raw_class.replace('_', ' ')

    # Determine if healthy or diseased
    is_healthy = 'healthy' in disease_name.lower()

    # Get top 3 predictions for display
    top3_indices = np.argsort(predictions[0])[-3:][::-1]
    top3 = []
    for idx in top3_indices:
        cls = class_labels[int(idx)]
        conf = float(predictions[0][idx]) * 100
        clean = cls.replace('___', ' - ').replace('_', ' ')
        top3.append({'class': clean, 'confidence': round(conf, 2)})

    # Get treatment recommendations
    recommendations = get_recommendations(disease_name, is_healthy)

    return {
        'plant_name': plant_name,
        'disease_name': disease_name,
        'is_healthy': is_healthy,
        'confidence': round(confidence, 2),
        'raw_class': raw_class,
        'top3': top3,
        'recommendations': recommendations
    }


def get_recommendations(disease_name, is_healthy):
    """Return treatment recommendations based on disease"""

    if is_healthy:
        return [
            'Continue regular watering and care',
            'Ensure adequate sunlight and nutrients',
            'Monitor for any changes in appearance',
            'Maintain good air circulation',
            'Keep leaves clean and dry'
        ]

    # Disease specific recommendations
    disease_lower = disease_name.lower()

    if 'blight' in disease_lower:
        return [
            'Remove and destroy infected plant parts immediately',
            'Apply copper-based fungicide every 7-10 days',
            'Avoid overhead watering to reduce moisture',
            'Improve air circulation around plants',
            'Rotate crops next season to prevent recurrence'
        ]
    elif 'rust' in disease_lower:
        return [
            'Remove infected leaves and dispose carefully',
            'Apply sulfur-based fungicide as soon as possible',
            'Avoid wetting foliage when watering',
            'Plant resistant varieties in future seasons',
            'Monitor surrounding plants for spread'
        ]
    elif 'scab' in disease_lower:
        return [
            'Apply fungicide at first sign of infection',
            'Remove fallen infected leaves from ground',
            'Prune trees to improve air circulation',
            'Use disease-resistant plant varieties',
            'Apply preventive spray during wet seasons'
        ]
    elif 'spot' in disease_lower:
        return [
            'Remove and destroy heavily infected leaves',
            'Apply appropriate fungicide treatment',
            'Water at the base to keep foliage dry',
            'Space plants properly for good air flow',
            'Avoid working with plants when wet'
        ]
    elif 'mold' in disease_lower or 'mildew' in disease_lower:
        return [
            'Improve ventilation around affected plants',
            'Apply potassium bicarbonate or neem oil',
            'Reduce humidity in growing environment',
            'Remove affected plant material carefully',
            'Avoid over-fertilizing with nitrogen'
        ]
    elif 'mosaic' in disease_lower or 'virus' in disease_lower:
        return [
            'Remove and destroy infected plants immediately',
            'Control aphids and insects that spread virus',
            'Disinfect garden tools after use',
            'Plant virus-resistant varieties next season',
            'There is no cure - prevention is key'
        ]
    elif 'bacterial' in disease_lower:
        return [
            'Apply copper-based bactericide spray',
            'Remove infected plant parts with sterile tools',
            'Avoid overhead irrigation',
            'Improve drainage to reduce waterlogging',
            'Disinfect tools between plants'
        ]
    else:
        return [
            'Isolate affected plants to prevent spread',
            'Consult an agricultural expert for diagnosis',
            'Consider appropriate fungicide treatment',
            'Monitor other plants for similar symptoms',
            'Improve overall plant growing conditions'
        ]

def analyze_weather_disease_correlation(weather_data, disease_name, is_healthy):
    """Analyze how weather might have contributed to the disease."""
    if is_healthy or not weather_data:
        return None

    disease_lower = disease_name.lower()
    
    # Extract weather values for logic
    try:
        temp = float(weather_data.get('temperature', '0').replace('°C', ''))
        humidity = float(weather_data.get('humidity', '0').replace('%', ''))
        wind = float(weather_data.get('wind_speed', '0').split(' ')[0])
        rain = float(weather_data.get('rainfall', '0').split(' ')[0])
    except ValueError:
        return None

    analysis = []
    
    # Check fungal conditions
    if any(x in disease_lower for x in ['blight', 'rust', 'scab', 'mold', 'mildew', 'spot']):
        if humidity > 70 and rain > 0:
            analysis.append("High humidity and recent rainfall have created highly favorable conditions for fungal diseases to thrive and spread.")
        elif humidity > 80:
            analysis.append("The current high humidity level creates a persistent moist environment that encourages fungal spore germination.")
            
    # Check bacterial conditions
    if 'bacterial' in disease_lower:
        if temp > 25 and humidity > 70:
            analysis.append("Warm temperatures combined with high moisture levels provide an ideal breeding ground for bacterial infections.")
            
    # Check wind spread
    if wind > 15 and not is_healthy:
        analysis.append("Current elevated wind speeds may actively contribute to spreading the pathogen to nearby healthy plants.")

    if not analysis:
        # Fallback explanation if no specific combination triggered
        analysis.append(f"Current environmental conditions (Temp: {weather_data.get('temperature')}, Humidity: {weather_data.get('humidity')}) are part of the microclimate affecting your plant's resilience to {disease_name}.")

    return " ".join(analysis)


# ============================================================
# ROUTES - Each route is one page of the website
# ============================================================

@app.route('/')
def home():
    """Home page"""
    return render_template('home.html')


@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')


@app.route('/upload')
def upload():
    """Upload page"""
    return render_template('upload.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle image upload and run prediction
    Called when user submits an image
    """
    # Check if file was included in request
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    # Check if file was actually selected
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Check if file format is allowed
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload JPG or PNG'}), 400

    if file:
        # Save file securely
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Run prediction
            prediction = predict_disease(filepath)

            # Save image to static folder for display
            static_filename = f"upload_{secrets.token_hex(8)}.jpg"
            static_path = os.path.join(STATIC_FOLDER, 'images', static_filename)
            Image.open(filepath).convert('RGB').save(static_path)
            
            # Weather Analysis part
            lat = request.form.get('lat')
            lon = request.form.get('lon')
            weather_data = None
            weather_analysis = None
            
            if lat and lon:
                # Fetch weather for prediction if coordinates were provided
                url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
                try:
                    w_res = requests.get(url, timeout=5)
                    if w_res.status_code == 200:
                        w_json = w_res.json()
                        weather_data = {
                            'location': w_json.get('name', 'Unknown Location'),
                            'temperature': f"{w_json['main']['temp']:.1f}°C",
                            'humidity': f"{w_json['main']['humidity']}%",
                            'wind_speed': f"{w_json['wind']['speed'] * 3.6:.1f} km/h",
                            'rainfall': f"{w_json.get('rain', {}).get('1h', 0)} mm"
                        }
                        weather_analysis = analyze_weather_disease_correlation(
                            weather_data, 
                            prediction['disease_name'], 
                            prediction['is_healthy']
                        )
                except Exception as e:
                    print(f"Weather error: {e}")

            # Store results in session
            session['prediction'] = prediction
            session['image_path'] = f"images/{static_filename}"
            session['weather_data'] = weather_data
            session['weather_analysis'] = weather_analysis

            # Clean up temp upload
            # os.remove(filepath)

            return jsonify({'success': True})

        except Exception as e:
            # Clean up on error
            if os.path.exists(filepath):
                # os.remove(filepath)
                pass
            return jsonify({'error': str(e)}), 500


@app.route('/result')
def result():
    """Result page - shows prediction"""
    prediction = session.get('prediction')
    image_path = session.get('image_path')
    weather_data = session.get('weather_data')
    weather_analysis = session.get('weather_analysis')

    # Redirect to upload if no prediction in session
    if not prediction:
        return redirect(url_for('upload'))

    return render_template('result.html',
                           prediction=prediction,
                           image_path=image_path,
                           weather_data=weather_data,
                           weather_analysis=weather_analysis)


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        context = data.get('context', {})
        language = data.get('language', 'English')

        plant_name = context.get('plant_name', 'Unknown')
        disease_name = context.get('disease_name', 'Unknown')
        confidence = context.get('confidence', 0)
        is_healthy = context.get('is_healthy', False)

        system_prompt = f"""You are PlantCare AI Assistant, 
an expert plant pathologist and agricultural specialist.

CURRENT DETECTION:
- Plant: {plant_name}
- Condition: {disease_name}
- Confidence: {confidence}%
- Status: {"Healthy" if is_healthy else "Disease Detected"}

YOUR ROLE:
- Answer questions specifically about this detected condition
- Give expert, practical, actionable agricultural advice
- Keep answers clear and concise (3-5 sentences)
- Use simple language farmers can understand
- Always relate back to the specific detected disease
- For treatments: give specific product names and dosages
- For spread: give realistic timeframes and risk conditions
- For food safety: be clear and direct
- End responses with one helpful tip or next step
- If asked off-topic: politely redirect to plant health

IMPORTANT REQUIREMENT:
- ALWAYS respond to the user in the {language} language.
- If {language} is Hindi or Telugu, use the native script.

You are talking to a farmer or gardener who needs help 
with their {plant_name} that has {disease_name}."""

        full_prompt = f"{system_prompt}\n\nQuestion: {user_message}\n\nExpert Answer in {language}:"

        response = gemini_model.generate_content(full_prompt)

        return jsonify({
            'success': True,
            'response': response.text
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/weather', methods=['POST'])
def api_weather():
    try:
        data = request.get_json()
        lat = data.get('lat')
        lon = data.get('lon')
        
        if not lat or not lon:
            return jsonify({'success': False, 'error': 'Missing coordinates'}), 400
            
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            w_json = response.json()
            weather = {
                'location': w_json.get('name', 'Unknown Location'),
                'temperature': f"{w_json['main']['temp']:.1f}°C",
                'humidity': f"{w_json['main']['humidity']}%",
                'wind_speed': f"{w_json['wind']['speed'] * 3.6:.1f} km/h",
                'rainfall': f"{w_json.get('rain', {}).get('1h', 0)} mm",
                'description': w_json['weather'][0]['description'].capitalize()
            }
            return jsonify({'success': True, 'data': weather})
            
        return jsonify({'success': False, 'error': 'Failed to fetch weather from API'}), response.status_code
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================
# RUN THE APP
# ============================================================
if __name__ == '__main__':
    # Load model before starting server
    load_model()

    # Create required folders if they don't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(os.path.join(STATIC_FOLDER, 'images'), exist_ok=True)

    print("\n" + "=" * 50)
    print("PlantCare AI Web App Starting...")
    print("=" * 50)
    print("Open your browser and go to:")
    print("   http://127.0.0.1:5000")
    print("=" * 50 + "\n")

    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)