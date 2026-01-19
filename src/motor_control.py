#!/usr/bin/env python3
import cv2, numpy as np, time, RPi.GPIO as GPIO
from auppbot import AUPPBot
from flask import Flask, Response
import threading
import onnxruntime as ort

# ============= CAMERA / ROI =============
CAM_INDEX = 0
W, H = 640, 480
ROTATE_90_CW = True

# ============= DIGIT DETECTION PARAMS =============
DIGIT_MODEL_PATH = "best.onnx"
MIN_DIGIT_CONFIDENCE = 0.3  # Lower threshold for initial testing
DIGIT_INPUT_SIZE = (320, 320)  # Model expects 320x320 RGB images
DIGIT_COOLDOWN = 0.5  # Seconds to wait after detecting same digit again
ALLOW_INTERRUPTS = True  # Allow new digits to interrupt current action

# ============= ULTRASONIC SENSOR =============
TRIG = 21
ECHO = 20

# ============= DRIVE PARAMS =============
PORT = "/dev/ttyUSB0"
BAUD = 115200
BASE = 13

LEFT_SIGN  = +1
RIGHT_SIGN = +1

MOTOR_SELF_TEST = True
SELF_TEST_PWM   = 12
SELF_TEST_SEC   = 0.8

# ============= DIGIT ACTION PARAMS =============
DIGIT_ACTION_DURATION = 1.5  # How long to execute digit actions
TURN_SPEED = 25
TURN_TIME_90 = 1.1  # Time to turn 90 degrees

# ============= WEB STREAMING =============
app = Flask(__name__)
current_frame = None
frame_lock = threading.Lock()

# ============= GLOBAL STATE =============
ultrasonic_distance = None
distance_lock = threading.Lock()
digit_session = None

# ============= SETUP GPIO =============
GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

def clamp99(x): 
    return int(max(-99, min(99, x)))

def set_tank(bot, left, right):
    try:
        l = clamp99(LEFT_SIGN  * left)
        r = clamp99(RIGHT_SIGN * right)
        bot.motor1.speed(l); bot.motor2.speed(l)
        bot.motor3.speed(r); bot.motor4.speed(r)
        print(f"  → Motors: L={l}, R={r}")
    except Exception as e:
        print(f"❌ Motor error: {e}")

def motor_self_test(bot):
    print("Motor self-test: TURN LEFT...")
    set_tank(bot, BASE - 5, BASE + 5)
    time.sleep(SELF_TEST_SEC)

    print("Motor self-test: TURN RIGHT...")
    set_tank(bot, BASE + 5, BASE - 5)
    time.sleep(SELF_TEST_SEC)

    print("Motor self-test: STRAIGHT...")
    set_tank(bot, BASE, BASE)
    time.sleep(SELF_TEST_SEC)

    print("Motor self-test: STOP")
    set_tank(bot, 0, 0)
    time.sleep(0.3)

def load_digit_model():
    """Load ONNX digit recognition model"""
    global digit_session
    try:
        digit_session = ort.InferenceSession(DIGIT_MODEL_PATH)
        print(f"✅ Digit model loaded: {DIGIT_MODEL_PATH}")
        
        # Print model info
        input_info = digit_session.get_inputs()[0]
        output_info = digit_session.get_outputs()[0]
        print(f"   Model input: {input_info.name}, shape: {input_info.shape}")
        print(f"   Model output: {output_info.name}, shape: {output_info.shape}")
        return True
    except Exception as e:
        print(f"❌ Failed to load digit model: {e}")
        return False

def preprocess_digit_region(region):
    """Preprocess image region for digit detection"""
    # Resize to model input size (keep RGB, 3 channels)
    resized = cv2.resize(region, DIGIT_INPUT_SIZE)
    
    # Normalize to [0, 1]
    normalized = resized.astype(np.float32) / 255.0
    
    # Transpose from HWC to CHW format: (height, width, channels) -> (channels, height, width)
    transposed = np.transpose(normalized, (2, 0, 1))
    
    # Add batch dimension: (channels, height, width) -> (batch, channels, height, width)
    input_tensor = np.expand_dims(transposed, axis=0)
    
    return input_tensor

def detect_digit(frame):
    """Detect digit anywhere in frame using YOLOv8. Returns (detected_digit, confidence)"""
    if digit_session is None:
        return None, 0.0
    
    try:
        # Use the entire frame for digit detection
        input_tensor = preprocess_digit_region(frame)
        
        # Get input name from model
        input_name = digit_session.get_inputs()[0].name
        
        # Run inference
        outputs = digit_session.run(None, {input_name: input_tensor})
        
        # YOLOv8 output format: (1, 4+num_classes, 8400)
        # Where first 4 are bounding box coords, rest are class probabilities
        output = outputs[0]  # Shape: (1, 4+num_classes, 8400)
        
        # Transpose to (8400, 4+num_classes)
        predictions = output[0].T  # Now shape: (8400, 4+num_classes)
        
        best_digit = None
        best_confidence = 0.0
        
        # Process each prediction
        for pred in predictions:
            # pred format: [x, y, w, h, class_0_conf, class_1_conf, ...]
            box = pred[:4]  # Bounding box coordinates
            class_scores = pred[4:]  # Class confidence scores
            
            # Get the class with highest confidence
            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])
            
            # Check if this is a digit we care about (0, 1, 2, 3, 4)
            if class_id in [0, 1, 2, 3, 4] and confidence > best_confidence:
                best_digit = class_id
                best_confidence = confidence
        
        # Return best detection if confidence is high enough
        if best_confidence >= MIN_DIGIT_CONFIDENCE:
            print(f"  🔍 Detected digit {best_digit} with confidence {best_confidence:.2%}")
            return best_digit, best_confidence
        
        return None, 0.0
        
    except Exception as e:
        print(f"⚠️ Digit detection error: {e}")
        import traceback
        traceback.print_exc()
        return None, 0.0

def execute_digit_action(bot, digit):
    """Execute action based on detected digit - returns immediately, action persists"""
    print("\n" + "="*60)
    print(f"🔢 DIGIT {digit} DETECTED - EXECUTING ACTION")
    print("="*60)
    
    if digit == 0:
        # Stop
        print("🛑 Digit 0: STOP")
        set_tank(bot, 0, 0)
        
    elif digit == 1:
        # Move forward
        print("➡️ Digit 1: MOVING FORWARD")
        set_tank(bot, BASE, BASE)
        
    elif digit == 2:
        # Move backward
        print("⬅️ Digit 2: MOVING BACKWARD")
        set_tank(bot, -BASE, -BASE)
        
    elif digit == 3:
        # Turn right
        print("↪️ Digit 3: TURNING RIGHT")
        set_tank(bot, TURN_SPEED, -TURN_SPEED)
        
    elif digit == 4:
        # Turn left
        print("↩️ Digit 4: TURNING LEFT")
        set_tank(bot, -TURN_SPEED, TURN_SPEED)
    
    print("✓ Action started (will continue until new digit detected)...")
    print("="*60 + "\n")

def read_ultrasonic():
    """Read ultrasonic sensor in a separate thread"""
    global ultrasonic_distance
    
    while True:
        try:
            GPIO.output(TRIG, False)
            time.sleep(0.05)

            GPIO.output(TRIG, True)
            time.sleep(0.00001)
            GPIO.output(TRIG, False)

            timeout = time.time() + 0.1
            pulse_start = time.time()
            while GPIO.input(ECHO) == 0:
                if time.time() > timeout:
                    break
                pulse_start = time.time()

            timeout = time.time() + 0.1
            pulse_end = time.time()
            while GPIO.input(ECHO) == 1:
                if time.time() > timeout:
                    break
                pulse_end = time.time()

            pulse_duration = pulse_end - pulse_start
            distance = pulse_duration * 17150
            
            with distance_lock:
                ultrasonic_distance = round(distance, 2)
            
            time.sleep(0.1)
            
        except Exception as e:
            print(f"⚠️ Ultrasonic error: {e}")
            time.sleep(0.1)

def generate_frames():
    """Generator for streaming frames"""
    global current_frame
    while True:
        with frame_lock:
            if current_frame is None:
                continue
            ret, buffer = cv2.imencode('.jpg', current_frame)
            frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '''
    <html>
      <head>
        <title>Robot - Digit Detection Control</title>
        <style>
          body { background: #000; display: flex; justify-content: center; 
                 align-items: center; height: 100vh; margin: 0; font-family: Arial; }
          .container { text-align: center; }
          h1 { color: #0f0; font-size: 24px; margin-bottom: 10px; }
          img { border: 3px solid #0f0; max-width: 95vw; }
          .info { color: #0f0; margin-top: 10px; font-size: 14px; }
          .legend { color: #0ff; margin-top: 15px; font-size: 13px; text-align: left; 
                    display: inline-block; }
        </style>
      </head>
      <body>
        <div class="container">
          <h1>🤖 Robot Digit Detection Control</h1>
          <img src="/video_feed" width="640" height="480">
          <div class="info">Show digit paper anywhere in camera view to control robot</div>
          <div class="legend">
            <strong>Digit Commands:</strong><br>
            • 0 = Stop<br>
            • 1 = Forward<br>
            • 2 = Backward<br>
            • 3 = Turn Right<br>
            • 4 = Turn Left
          </div>
        </div>
      </body>
    </html>
    '''

def robot_control_loop(bot, cap):
    """Main control loop: ONLY digit detection (no line following)"""
    global current_frame, ultrasonic_distance
    
    last_print = 0.0
    
    # Digit detection state
    last_digit_detection = 0
    digit_cooldown_active = False
    last_executed_digit = None  # Track last executed digit
    
    # Robot starts idle (stopped)
    set_tank(bot, 0, 0)
    
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("❌ Failed to grab frame")
                break

            if ROTATE_90_CW:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            frame = cv2.resize(frame, (W, H))
            h, w = frame.shape[:2]

            # ===== DIGIT DETECTION (Continuous check) =====
            detected_digit = None
            digit_confidence = 0.0
            
            current_time = time.time()
            if current_time - last_digit_detection > DIGIT_COOLDOWN:
                digit_cooldown_active = False
            
            if not digit_cooldown_active:
                detected_digit, digit_confidence = detect_digit(frame)
                
                if detected_digit is not None:
                    print(f"🔢 Digit {detected_digit} detected with {digit_confidence:.2%} confidence")
                    execute_digit_action(bot, detected_digit)
                    last_digit_detection = time.time()
                    digit_cooldown_active = True
                    last_executed_digit = detected_digit  # Store executed digit

            # ===== ULTRASONIC READING (for display only) =====
            dist = None
            with distance_lock:
                dist = ultrasonic_distance

            # ===== VISUALIZATION =====
            vis = frame.copy()
            
            # Show detected digit
            if detected_digit is not None:
                digit_text = f"DIGIT: {detected_digit}"
                conf_text = f"Confidence: {digit_confidence:.1%}"
                cv2.putText(vis, digit_text, (w//2 - 80, h//2 - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                cv2.putText(vis, conf_text, (w//2 - 100, h//2 + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Digit cooldown indicator
            if digit_cooldown_active:
                cooldown_remaining = DIGIT_COOLDOWN - (time.time() - last_digit_detection)
                cv2.rectangle(vis, (5, 5), (250, 60), (50, 50, 50), -1)
                cv2.putText(vis, f"Cooldown: {cooldown_remaining:.1f}s", (10, 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
            else:
                cv2.rectangle(vis, (5, 5), (250, 60), (50, 50, 50), -1)
                cv2.putText(vis, "READY", (10, 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Status display
            status_text = "IDLE - Waiting for digit..."
            if digit_cooldown_active and last_executed_digit is not None:
                status_text = f"Executed: Digit {last_executed_digit}"
            
            cv2.rectangle(vis, (5, h - 80), (w - 5, h - 5), (50, 50, 50), -1)
            cv2.putText(vis, status_text, (10, h - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Ultrasonic distance (for reference)
            if dist is not None:
                cv2.putText(vis, f"Distance: {dist:.1f}cm", (10, h - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            with frame_lock:
                current_frame = vis.copy()

            # Throttled print
            if time.time() - last_print > 1.0:
                digit_status = f"Digit={detected_digit} ({digit_confidence:.1%})" if detected_digit else "No digit"
                cooldown_status = "COOLDOWN" if digit_cooldown_active else "READY"
                print(f"[{cooldown_status}] [{digit_status}] [Dist: {dist}cm]")
                last_print = time.time()

    finally:
        try: 
            set_tank(bot, 0, 0)
            bot.stop_all()
        except: 
            pass

def main():
    print("🚀 Starting robot initialization...")
    
    # Load digit detection model
    if not load_digit_model():
        print("❌ ERROR: Cannot run without digit detection model!")
        return
    
    try:
        bot = AUPPBot(PORT, BAUD, auto_safe=True)
        print("✅ Robot connected!")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        return
    
    if MOTOR_SELF_TEST:
        motor_self_test(bot)

    print("Opening camera...")
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    print("✅ Camera opened!")

    try:
        bot.servo1.angle(40)
        print("✅ Servo set to 40°")
    except Exception as e:
        print(f"⚠️ Servo error: {e}")

    # Start ultrasonic sensor thread
    ultrasonic_thread = threading.Thread(target=read_ultrasonic, daemon=True)
    ultrasonic_thread.start()
    print("✅ Ultrasonic sensor started")

    # Start robot control thread
    control_thread = threading.Thread(target=robot_control_loop, args=(bot, cap), daemon=True)
    control_thread.start()

    import socket
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    
    print("\n" + "="*70)
    print("🎥 DIGIT DETECTION CONTROL SYSTEM STARTED!")
    print("="*70)
    print(f"\n📱 Open your browser and go to:")
    print(f"\n   http://{ip_address}:5000")
    print(f"\n   or try: http://raspberrypi.local:5000")
    print("\n" + "="*70)
    print("Control your robot with digit papers:")
    print("\n  0 = Stop")
    print("  1 = Move Forward")
    print("  2 = Move Backward") 
    print("  3 = Turn Right")
    print("  4 = Turn Left")
    print("\nHold digit paper anywhere in camera view")
    print("Press Ctrl+C to stop\n")

    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping robot...")
    finally:
        try: 
            set_tank(bot, 0, 0)
            bot.stop_all()
            bot.close()
        except: 
            pass
        cap.release()
        GPIO.cleanup()

if __name__ == "__main__":
    main()