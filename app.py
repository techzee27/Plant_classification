# ============================================================
# app.py - Dr. Groot Ai Flask Backend
# ============================================================

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_file
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import tensorflow as tf
import json
import os
import secrets
import google.generativeai as genai
import requests
from dotenv import load_dotenv

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm, cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, 
    Table, TableStyle, Image as RLImage, HRFlowable, Flowable, PageBreak)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.graphics import renderPDF
from io import BytesIO
from datetime import datetime
import random

load_dotenv()

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

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY is not set in environment or .env file.")

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.5-flash')

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
if not OPENWEATHER_API_KEY:
    print("WARNING: OPENWEATHER_API_KEY is not set in environment or .env file.")

# ============================================================
# LOAD MODEL AND CLASS LABELS
# ============================================================
model = None
class_labels = None
class_indices = None

def load_model():
    """Load the trained model and class labels"""
    global model, class_labels, class_indices

    print("Loading Dr. Groot Ai model...")

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


def get_disease_info(disease_name):
    d = disease_name.lower()
    if 'late blight' in d:
        return {'type': 'Fungal', 'temp_range': '10-25°C', 'humidity_risk': '>90% humidity critical - disease spreads rapidly', 'rain_risk': 'Wet weather critical - avoid overhead irrigation', 'spread_speed': 'RAPID', 'spread_days': '7-10 days', 'yield_loss': 'Up to 100% possible', 'urgency': 'CRITICAL', 'causal_agent': 'Phytophthora infestans', 'host_plants': 'Tomato, Potato'}
    elif 'early blight' in d:
        return {'type': 'Fungal', 'temp_range': '24-29°C', 'humidity_risk': 'Moderate humidity (60-80%) favors spread', 'rain_risk': 'Rain splash spreads spores between plants', 'spread_speed': 'MODERATE', 'spread_days': '2-3 weeks', 'yield_loss': '20-40%', 'urgency': 'HIGH', 'causal_agent': 'Alternaria solani', 'host_plants': 'Tomato, Potato'}
    elif 'scab' in d:
        return {'type': 'Fungal', 'temp_range': '15-25°C', 'humidity_risk': 'High humidity with leaf wetness required', 'rain_risk': 'Spring rain spreads spores', 'spread_speed': 'MODERATE', 'spread_days': '1-3 weeks', 'yield_loss': 'Decreased market value', 'urgency': 'HIGH', 'causal_agent': 'Venturia inaequalis', 'host_plants': 'Apple, Pear'}
    elif 'rust' in d:
        return {'type': 'Fungal', 'temp_range': '15-25°C', 'humidity_risk': '>95% humidity with leaf wetness required', 'rain_risk': 'Heavy dew and rain required for infection', 'spread_speed': 'FAST', 'spread_days': 'days', 'yield_loss': '30-50% in severe cases', 'urgency': 'HIGH', 'causal_agent': 'Puccinia spp.', 'host_plants': 'Corn, Wheat, etc.'}
    elif 'powdery mildew' in d:
        return {'type': 'Fungal', 'temp_range': '20-27°C', 'humidity_risk': 'Low-moderate humidity (40-70%) favors this disease', 'rain_risk': 'Ironically worsened by dry spells', 'spread_speed': 'MODERATE', 'spread_days': '1-2 weeks', 'yield_loss': 'Moderate to severe if unchecked', 'urgency': 'MODERATE', 'causal_agent': 'Various Erysiphaceae', 'host_plants': 'Many plants'}
    elif 'bacterial spot' in d or 'bacterial' in d:
        return {'type': 'Bacterial', 'temp_range': '25-30°C', 'humidity_risk': 'High humidity and rain splash spreads bacteria', 'rain_risk': 'Rain and flooding spread bacterial pathogens', 'spread_speed': 'MODERATE', 'spread_days': '1-2 weeks', 'yield_loss': '15-30%', 'urgency': 'HIGH', 'causal_agent': 'Xanthomonas spp.', 'host_plants': 'Tomato, Pepper'}
    elif 'mosaic' in d or 'virus' in d:
        return {'type': 'Viral', 'temp_range': 'Any', 'humidity_risk': 'Not direct (indirect via insect vectors)', 'rain_risk': 'Minimal direct effect', 'spread_speed': 'SLOW', 'spread_days': 'Permanent', 'yield_loss': 'High', 'urgency': 'HIGH', 'causal_agent': 'Various Viruses', 'host_plants': 'Many plants'}
    elif 'healthy' in d:
        return {'type': 'Healthy', 'temp_range': 'Current temperature range is safe', 'humidity_risk': 'Maintain good air circulation to prevent disease', 'rain_risk': 'Avoid waterlogging, ensure good drainage', 'spread_speed': 'NONE', 'spread_days': 'N/A', 'yield_loss': 'No yield loss risk detected', 'urgency': 'NONE', 'causal_agent': 'None', 'host_plants': 'N/A'}
    else:
        return {'type': 'Unknown', 'temp_range': 'Unknown', 'humidity_risk': 'Unknown', 'rain_risk': 'Unknown', 'spread_speed': 'MODERATE', 'spread_days': 'Unknown', 'yield_loss': 'Unknown', 'urgency': 'MODERATE', 'causal_agent': 'Unknown pathogen', 'host_plants': 'Unknown'}

def draw_confidence_bar(canvas, x, y, width, height, percentage):
    canvas.setFillColorRGB(0.9, 0.9, 0.9)
    canvas.rect(x, y, width, height, fill=1, stroke=0)
    fill_width = width * (percentage / 100)
    if percentage > 80:
        canvas.setFillColorRGB(0.09, 0.64, 0.29)
    elif percentage > 60:
        canvas.setFillColorRGB(0.95, 0.61, 0.07)
    else:
        canvas.setFillColorRGB(0.94, 0.27, 0.27)
    canvas.rect(x, y, fill_width, height, fill=1, stroke=0)

class ConfBarFlowable(Flowable):
    def __init__(self, w, h, p):
        Flowable.__init__(self)
        self.width = w
        self.height = h
        self.percentage = p
    def draw(self):
        draw_confidence_bar(self.canv, 0, 0, self.width, self.height, self.percentage)

class HeaderBanner(Flowable):
    def __init__(self, width, height, date_str, id_str):
        Flowable.__init__(self)
        self.width = width
        self.height = height
        self.date_str = date_str
        self.id_str = id_str
    def draw(self):
        c = self.canv
        c.setFillColorRGB(22/255.0, 163/255.0, 74/255.0)
        c.rect(0, 0, self.width, self.height, fill=1, stroke=0)
        c.setFillColorRGB(1, 1, 1)
        c.setFont("Helvetica-Bold", 24)
        c.drawString(20, self.height - 35, "Dr. Groot Ai")
        c.setFont("Helvetica", 11)
        c.setFillColorRGB(0.8, 1, 0.8)
        c.drawString(20, self.height - 55, "Advanced Plant Disease Detection System")
        c.setFillColorRGB(1, 1, 1)
        c.setFont("Helvetica-Bold", 14)
        c.drawRightString(self.width - 20, self.height - 30, "DIAGNOSTIC REPORT")
        c.setFont("Helvetica", 10)
        c.drawRightString(self.width - 20, self.height - 45, f"Report ID: {self.id_str}")
        c.drawRightString(self.width - 20, self.height - 60, f"Date: {self.date_str}")

class FooterBanner(Flowable):
    def __init__(self, w, h, id_str):
        Flowable.__init__(self)
        self.width = w
        self.height = h
        self.id_str = id_str
    def draw(self):
        c = self.canv
        c.setStrokeColorRGB(22/255.0, 163/255.0, 74/255.0)
        c.line(0, self.height, self.width, self.height)
        c.setFont("Helvetica", 9)
        c.setFillColorRGB(0.4, 0.4, 0.4)
        c.drawString(0, self.height - 15, "Generated by Dr. Groot Ai | Powered by MobileNetV2")
        c.drawCentredString(self.width/2.0, self.height - 15, f"Report ID: {self.id_str} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        c.drawRightString(self.width, self.height - 15, "Disclaimer: AI diagnosis should be confirmed by certified expert")

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


@app.route('/download-report')
def download_report():
    prediction = session.get('prediction')
    image_path = session.get('image_path')
    
    if not prediction:
        return redirect(url_for('upload'))
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=15*mm, leftMargin=15*mm, topMargin=10*mm, bottomMargin=10*mm)
    elements = []
    
    styles = getSampleStyleSheet()
    style_normal = styles["Normal"]
    style_normal.fontName = 'Helvetica'
    style_title = ParagraphStyle('Title', parent=style_normal, fontName='Helvetica-Bold', fontSize=12, textColor=colors.HexColor("#16a34a"))
    style_subtitle = ParagraphStyle('SubTitle', parent=style_normal, fontName='Helvetica-Bold', fontSize=11, spaceAfter=8, textColor=colors.HexColor("#1a1a1a"))
    style_center = ParagraphStyle('Center', parent=style_normal, alignment=TA_CENTER)
    
    report_id = f"PC-{random.randint(100000, 999999)}"
    report_date = datetime.now().strftime("%B %d, %Y")
    
    elements.append(HeaderBanner(doc.width, 80, report_date, report_id))
    elements.append(Spacer(1, 10))
    
    t_info = Table([[
        Paragraph(f"<b>Report Date</b><br/>{datetime.now().strftime('%d/%m/%Y')}", style_center),
        Paragraph(f"<b>Report Time</b><br/>{datetime.now().strftime('%H:%M:%S')}", style_center),
        Paragraph("<b>Analysis Method</b><br/>MobileNetV2", style_center),
        Paragraph("<b>Report Status</b><br/><font color='green'>COMPLETED ✓</font>", style_center)
    ]], colWidths=[doc.width/4.0]*4)
    t_info.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f3f4f6")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e5e7eb")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#e5e7eb")),
        ('PADDING', (0,0), (-1,-1), 5)
    ]))
    elements.append(t_info)
    elements.append(Spacer(1, 15))
    
    plant_name = prediction.get('plant_name', 'Unknown')
    disease_name = prediction.get('disease_name', 'Unknown')
    confidence = prediction.get('confidence', 0)
    is_healthy = prediction.get('is_healthy', False)
    
    box_color = colors.HexColor("#22c55e") if is_healthy else colors.HexColor("#ef4444")
    status_text = "✓ HEALTHY PLANT" if is_healthy else "⚠ DISEASE DETECTED"
    conf_level = "VERY HIGH CONFIDENCE" if confidence > 90 else "HIGH CONFIDENCE" if confidence > 75 else "MODERATE CONFIDENCE" if confidence > 60 else "LOW CONFIDENCE - Manual verification recommended"
    
    left_cell = [
        Paragraph("PLANT SPECIES", style_normal),
        Paragraph(f"<b>{plant_name}</b>", ParagraphStyle('P1', fontName='Helvetica-Bold', fontSize=18)),
        Spacer(1, 5),
        Paragraph("DETECTED CONDITION", style_normal),
        Paragraph(f"<b><font color='{box_color}'>{disease_name}</font></b>", ParagraphStyle('P2', fontName='Helvetica-Bold', fontSize=16)),
        Spacer(1, 5),
        Paragraph(f"<b><font color='white'>{status_text}</font></b>", ParagraphStyle('Badge', backColor=box_color, alignment=TA_CENTER, borderPadding=3))
    ]
    right_cell = [
        Paragraph("CONFIDENCE SCORE", style_normal),
        Paragraph(f"<b><font color='#22c55e'>{confidence}%</font></b>", ParagraphStyle('P3', fontName='Helvetica-Bold', fontSize=24)),
        Spacer(1, 5),
        ConfBarFlowable(120, 15, confidence),
        Spacer(1, 5),
        Paragraph(conf_level, ParagraphStyle('P4', fontName='Helvetica', fontSize=9))
    ]
    t_diag = Table([[left_cell, right_cell]], colWidths=[doc.width/2.0, doc.width/2.0])
    t_diag.setStyle(TableStyle([('BOX', (0,0), (-1,-1), 2, box_color), ('VALIGN', (0,0), (-1,-1), 'TOP'), ('PADDING', (0,0), (-1,-1), 8)]))
    elements.append(Paragraph("<b>PRIMARY DIAGNOSIS</b>", style_title))
    elements.append(t_diag)
    elements.append(Spacer(1, 15))
    
    elements.append(Paragraph("<b>ANALYZED IMAGE</b>", style_subtitle))
    if image_path:
        img_full_path = os.path.join(STATIC_FOLDER, image_path)
        if os.path.exists(img_full_path):
            try:
                img = RLImage(img_full_path, width=150, height=100, kind='proportional')
                img.hAlign = 'CENTER'
                elements.append(img)
            except:
                pass
    elements.append(Paragraph("Image analyzed by MobileNetV2 CNN", style_center))
    elements.append(Spacer(1, 15))
    
    elements.append(Paragraph("<b>AI PREDICTION ANALYSIS</b>", style_subtitle))
    if 'top3' in prediction:
        top3_data = [["Rank", "Disease/Condition", "Confidence", "Bar"]]
        for i, item in enumerate(prediction['top3']):
            rank = ["#1", "#2", "#3"][i] if i < 3 else f"#{i+1}"
            top3_data.append([
                Paragraph(rank, style_normal), 
                Paragraph(item['class'], style_normal), 
                f"{item['confidence']}%",
                ConfBarFlowable(60, 10, item['confidence'])
            ])
        t_top3 = Table(top3_data, colWidths=[40, doc.width-180, 50, 90])
        t_top3.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f3f4f6")),
            ('BACKGROUND', (0,1), (-1,1), colors.HexColor("#dcffe4")),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e5e7eb")),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE')
        ]))
        elements.append(t_top3)
    elements.append(Spacer(1, 15))
    
    disease_info = get_disease_info(disease_name)
    elements.append(Paragraph("<b>WEATHER & ENVIRONMENTAL RISK ASSESSMENT</b>", style_subtitle))
    
    current_month = datetime.now().month
    if current_month in [6,7,8,9]: season_risk = "CRITICAL RISK (Monsoon)"
    elif current_month in [10,11,1,2]: season_risk = "MODERATE RISK (Winter)"
    else: season_risk = "LOW RISK (Summer)"
    
    w_data = [
        [Paragraph(f"<b>Temperature Risk:</b><br/>{disease_info['temp_range']}", style_normal), Paragraph(f"<b>Humidity Risk:</b><br/>{disease_info['humidity_risk']}", style_normal)],
        [Paragraph(f"<b>Rainfall Risk:</b><br/>{disease_info['rain_risk']}", style_normal), Paragraph(f"<b>Seasonal Risk:</b><br/>{season_risk}", style_normal)],
        [Paragraph(f"<b>Spread Speed:</b><br/>{disease_info['spread_speed']}", style_normal), Paragraph(f"<b>Overall Risk:</b><br/><font color='red'>{disease_info['urgency']}</font>", style_normal)]
    ]
    t_weather = Table(w_data, colWidths=[doc.width/2.0]*2)
    t_weather.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#e5e7eb")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e5e7eb")),
        ('PADDING', (0,0), (-1,-1), 6)
    ]))
    elements.append(t_weather)
    elements.append(Spacer(1, 15))
    
    elements.append(Paragraph("<b>RECOMMENDED TREATMENT PLAN</b>", style_subtitle))
    for i, rec in enumerate(prediction.get('recommendations', [])):
        urgency = "Immediate (24-48 hours)" if i == 0 and not is_healthy else "Ongoing"
        t_rec = Table([[Paragraph(f"<b>{(i+1)}.</b>", style_center), Paragraph(f"<b>Step {i+1}</b> - <i>{urgency}</i><br/>{rec}", style_normal)]], colWidths=[20, doc.width-20])
        t_rec.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'TOP')]))
        elements.append(t_rec)
        elements.append(Spacer(1, 5))
    elements.append(Spacer(1, 10))
    
    elements.append(Paragraph("<b>SEASONAL PREVENTION SCHEDULE</b>", style_subtitle))
    cal_data = [
        [Paragraph("<b>Season</b>", style_normal), Paragraph("<b>Risk</b>", style_normal), Paragraph("<b>Actions</b>", style_normal), Paragraph("<b>Products</b>", style_normal)],
        [Paragraph("Spring", style_normal), Paragraph("Medium", style_normal), Paragraph("Preventive spray, soil prep", style_normal), Paragraph("Copper fungicide", style_normal)],
        [Paragraph("Summer", style_normal), Paragraph("High", style_normal), Paragraph("Monitoring, irrigation mgmt", style_normal), Paragraph("Mancozeb", style_normal)],
        [Paragraph("Monsoon", style_normal), Paragraph("Critical", style_normal), Paragraph("Daily inspection, drainage", style_normal), Paragraph("Chlorothalonil", style_normal)],
        [Paragraph("Winter", style_normal), Paragraph("Low", style_normal), Paragraph("Pruning, sanitation", style_normal), Paragraph("Sulfur dust", style_normal)]
    ]
    t_cal = Table(cal_data, colWidths=[doc.width*0.15, doc.width*0.15, doc.width*0.4, doc.width*0.3])
    t_cal.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f3f4f6")), ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e5e7eb")), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
    elements.append(t_cal)
    
    elements.append(FooterBanner(doc.width, 30, report_id))
    elements.append(PageBreak())
    
    elements.append(Paragraph("<b>Dr. Groot Ai - Detailed Analysis Report</b> (Page 2)", ParagraphStyle('P2H', parent=style_subtitle, textColor=colors.HexColor("#16a34a"))))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#16a34a"), spaceBefore=5, spaceAfter=15))
    
    elements.append(Paragraph("<b>ABOUT THIS DISEASE</b>", style_subtitle))
    d_info_data = [
        [Paragraph(f"<b>Classification:</b> {disease_info.get('type', 'Unknown')}", style_normal), Paragraph(f"<b>First Symptoms:</b> Spots or discoloration", style_normal)],
        [Paragraph(f"<b>Causal Agent:</b> {disease_info.get('causal_agent', 'Unknown')}", style_normal), Paragraph(f"<b>Advanced Stage:</b> Severe necrosis", style_normal)],
        [Paragraph(f"<b>Host Plants:</b> {disease_info.get('host_plants', 'Unknown')}", style_normal), Paragraph(f"<b>Diagnostic Confusion:</b> Nutrient deficiency", style_normal)]
    ]
    t_dinfo = Table(d_info_data, colWidths=[doc.width/2.0]*2)
    t_dinfo.setStyle(TableStyle([('BOX', (0,0), (-1,-1), 1, colors.HexColor("#e5e7eb")), ('PADDING', (0,0), (-1,-1), 6)]))
    elements.append(t_dinfo)
    elements.append(Spacer(1, 15))
    
    elements.append(Paragraph("<b>POTENTIAL ECONOMIC IMPACT</b>", style_subtitle))
    econ_data = [[
        Paragraph(f"<b>Yield Loss Risk</b><br/>{disease_info.get('yield_loss', '')}", style_normal),
        Paragraph("<b>Treatment Cost</b><br/>₹800-2000 per acre", style_normal),
        Paragraph(f"<b>Urgency Level</b><br/><font color='red'>{disease_info.get('urgency', '')}</font>", style_normal)
    ]]
    t_econ = Table(econ_data, colWidths=[doc.width/3.0]*3)
    t_econ.setStyle(TableStyle([('BOX', (0,0), (-1,-1), 1, colors.HexColor("#e5e7eb")), ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e5e7eb")), ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('PADDING', (0,0), (-1,-1), 8)]))
    elements.append(t_econ)
    elements.append(Spacer(1, 15))
    
    elements.append(Paragraph("<b>TREATMENT OPTIONS COMPARISON</b>", style_subtitle))
    treat_comp = [
        [Paragraph("<b>Treatment Type</b>", style_normal), Paragraph("<b>Products</b>", style_normal), Paragraph("<b>Effectiveness</b>", style_normal), Paragraph("<b>Notes</b>", style_normal)],
        [Paragraph("Organic", style_normal), Paragraph("Neem oil, Copper", style_normal), Paragraph("60-75%", style_normal), Paragraph("Safe, slow acting", style_normal)],
        [Paragraph("Chemical Fungicide", style_normal), Paragraph("Mancozeb", style_normal), Paragraph("85-95%", style_normal), Paragraph("Fast, wear PPE", style_normal)],
        [Paragraph("Biological", style_normal), Paragraph("Bacillus subtilis", style_normal), Paragraph("70-80%", style_normal), Paragraph("Eco-friendly", style_normal)],
        [Paragraph("Preventive", style_normal), Paragraph("Crop rotation", style_normal), Paragraph("90%+", style_normal), Paragraph("Best long-term", style_normal)]
    ]
    t_comp = Table(treat_comp, colWidths=[doc.width*0.25, doc.width*0.3, doc.width*0.2, doc.width*0.25])
    t_comp.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f3f4f6")), ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e5e7eb")), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
    elements.append(t_comp)
    elements.append(Spacer(1, 20))
    
    code_cell = [
        Paragraph("<b>SCAN FOR MORE INFORMATION</b>", style_center), Spacer(1, 5),
        Paragraph("[ QR Code Placeholder ]", style_center), Spacer(1, 5),
        Paragraph("Visit drgrootai.com for detailed guides", style_center)
    ]
    t_qr = Table([[code_cell]], colWidths=[doc.width])
    t_qr.setStyle(TableStyle([('BOX', (0,0), (-1,-1), 1, colors.HexColor("#e5e7eb")), ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('PADDING', (0,0), (-1,-1), 10)]))
    elements.append(t_qr)
    elements.append(Spacer(1, 40))
    elements.append(FooterBanner(doc.width, 30, report_id))
    
    try:
        doc.build(elements)
    except Exception as e:
        print(f"PDF Build Error: {e}")
        
    buffer.seek(0)
    date_str2 = datetime.now().strftime("%Y%m%d")
    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"Dr_Groot_Report_{prediction['plant_name']}_{date_str2}.pdf",
        mimetype='application/pdf'
    )


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

        system_prompt = f"""You are Dr. Groot Ai Assistant, 
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
    print("Dr. Groot Ai Web App Starting...")
    print("=" * 50)
    print("Open your browser and go to:")
    print("   http://127.0.0.1:5000")
    print("=" * 50 + "\n")

    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)