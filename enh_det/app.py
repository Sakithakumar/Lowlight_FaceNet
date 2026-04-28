from flask import Flask, render_template, request, redirect, url_for, session, flash, Response
from werkzeug.security import generate_password_hash, check_password_hash
import os
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import threading
import time
import json

# Import your existing recognition code
from low_light_recognition import recognize_faces, load_known_faces

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Create Flask app with absolute paths
app = Flask(__name__, 
            template_folder=os.path.join(current_dir, 'templates'),
            static_folder=os.path.join(current_dir, 'static'))
app.secret_key = 'your_secret_key_here'

# Print paths for debugging
print(f"Current directory: {current_dir}")
print(f"Template folder: {app.template_folder}")
print(f"Static folder: {app.static_folder}")

# Check if template files exist
dashboard_path = os.path.join(app.template_folder, 'dashboard.html')
print(f"Dashboard template exists: {os.path.exists(dashboard_path)}")

# Global variables for detection control
detection_active = False
detection_thread = None
latest_frame = None
frame_lock = threading.Lock()

# Create a simple in-memory user database
users = {}

# Load known faces at startup
known_faces = load_known_faces()

# Add a route to check if templates are accessible
@app.route('/check_templates')
def check_templates():
    template_dir = app.template_folder
    template_path = os.path.join(os.getcwd(), template_dir)
    
    # List all files in the template directory
    try:
        files = os.listdir(template_path)
        file_list = "<br>".join(files)
        return f"Template directory: {template_path}<br>Files:<br>{file_list}"
    except Exception as e:
        return f"Error accessing template directory: {str(e)}"

@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username in users and check_password_hash(users[username], password):
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        if username in users:
            flash('Username already exists!', 'danger')
        elif password != confirm_password:
            flash('Passwords do not match!', 'danger')
        else:
            users[username] = generate_password_hash(password)
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    # Check if template exists before rendering
    template_path = os.path.join(app.template_folder, 'dashboard.html')
    if not os.path.exists(template_path):
        return f"Template not found at: {template_path}"
    
    return render_template('dashboard.html', username=session['username'])

@app.route('/detect')
def detect():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    # Check if template exists before rendering
    template_path = os.path.join(app.template_folder, 'detect.html')
    if not os.path.exists(template_path):
        return f"Template not found at: {template_path}"
    
    return render_template('detect.html', username=session['username'])

@app.route('/start_detection')
def start_detection():
    global detection_active, detection_thread
    
    if not detection_active:
        detection_active = True
        detection_thread = threading.Thread(target=detection_loop)
        detection_thread.daemon = True
        detection_thread.start()
        print("✅ Detection started")
        return {'status': 'success', 'message': 'Detection started'}
    
    return {'status': 'info', 'message': 'Detection already running'}

@app.route('/stop_detection')
def stop_detection():
    global detection_active
    if detection_active:
        detection_active = False
        # Wait for the thread to finish
        if detection_thread and detection_thread.is_alive():
            detection_thread.join(timeout=1.0)
        print("✅ Detection stopped")
    return {'status': 'success', 'message': 'Detection stopped'}

def detection_loop():
    global detection_active, latest_frame
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        detection_active = False
        return
    
    print("✅ Webcam opened successfully")
    
    try:
        while detection_active:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Failed to read frame")
                break
            
            frame = cv2.flip(frame, 1)
            results, enhanced_frame = recognize_faces(frame, known_faces, threshold=0.4)
            
            # Draw results
            display_frame = enhanced_frame.copy()
            for res in results:
                x1, y1, x2, y2 = res['bbox']
                name = res['name']
                conf = res['confidence']
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                label = f"{name} ({conf:.2f})" if name != "Unknown" else "Unknown"
                cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Convert frame to base64 for web display
            _, buffer = cv2.imencode('.jpg', display_frame)
            jpg_as_text = base64.b64encode(buffer)
            
            # Update the global latest_frame with thread safety
            with frame_lock:
                latest_frame = jpg_as_text.decode('utf-8')
            
            # Small delay to reduce CPU usage
            time.sleep(0.03)
    except Exception as e:
        print(f"❌ Error in detection loop: {str(e)}")
    finally:
        cap.release()
        print("✅ Webcam released")

@app.route('/get_frame')
def get_frame():
    global latest_frame
    with frame_lock:
        if latest_frame is not None:
            return latest_frame
    return ""

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    global detection_active, latest_frame
    
    while True:
        if detection_active and latest_frame:
            # Convert base64 to image
            img_data = base64.b64decode(latest_frame)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is not None:
                # Encode frame as JPEG
                ret, jpeg = cv2.imencode('.jpg', img)
                frame = jpeg.tobytes()
                
                # Yield the frame in the required format
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        
        time.sleep(0.03)

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

if __name__ == '__main__':
    # Create directories if they don't exist
    os.makedirs(os.path.join(current_dir, 'templates'), exist_ok=True)
    os.makedirs(os.path.join(current_dir, 'static'), exist_ok=True)
    os.makedirs(os.path.join(current_dir, 'static', 'css'), exist_ok=True)
    os.makedirs(os.path.join(current_dir, 'static', 'js'), exist_ok=True)
    os.makedirs(os.path.join(current_dir, 'static', 'images'), exist_ok=True)
    os.makedirs(os.path.join(current_dir, 'known_faces'), exist_ok=True)
    
    # Print startup message
    print("\n" + "="*50)
    print("Starting Low-Light Face Recognition Web App")
    print("="*50)
    print(f"Template folder: {app.template_folder}")
    print(f"Static folder: {app.static_folder}")
    print(f"Known faces folder: {os.path.join(current_dir, 'known_faces')}")
    print("="*50)
    print("App running at: http://localhost:5000")
    print("Check templates at: http://localhost:5000/check_templates")
    print("="*50 + "\n")
    
    app.run(debug=True, threaded=True)