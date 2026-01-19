import cv2
import numpy as np
import onnxruntime as ort

# --- CONFIGURATION ---
MODEL_PATH = 'best.onnx' # Ensure this file is in the same folder
INPUT_SIZE = 320 # Must match the 'imgsz' you used during export
CONF_THRESHOLD = 0.7
IOU_THRESHOLD = 0.45
CLASS_NAMES = ['0', '1', '2', '3', '4'] # Only 5 classes: indices 0 to 4

# --- INITIALIZATION ---
try:
    # Load the ONNX model session using ONNX Runtime
    session = ort.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
except Exception as e:
    print(f"Error loading ONNX model: {e}")
    exit()

# Open the default webcam (0 is usually the built-in webcam)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

def postprocess(raw_output, frame, conf_threshold, iou_threshold):
    # Fix 1: Correctly access the output tensor data for YOLOv8 ONNX
    output = raw_output[0][0].transpose() # Accesses the content of the single-item batch 

    # Insert this line right here (before the 'boxes, confs, class_ids = [], [], []' line):
    print(f"Post-processed tensor shape: {output.shape}")
    
    boxes, confs, class_ids = [], [], []
    frame_h, frame_w = frame.shape[:2]
    
    # Scale factors to convert normalized (320x320) coordinates back to original frame size
    x_scale = frame_w / INPUT_SIZE
    y_scale = frame_h / INPUT_SIZE
    
    for row in output:
        # The first 4 values are bounding box (cx, cy, w, h).
        # The remaining 5 values (row[4] onwards) are the Class Scores for 0-4.
        
        # 1. Identify the Class Scores slice (starts at index 4)
        class_scores = row[4:]
        
        # 2. Find the highest score and its index (class_id)
        raw_class_id = np.argmax(class_scores)
        score = class_scores[raw_class_id] # The score itself is our confidence
        
        confidence = score 

        # --- CRITICAL CHECK: Filter by the score/confidence threshold ---
        if confidence >= conf_threshold:
            
            # 3. Validate Class ID: Ensure the index is safe (0 to 4)
            class_id = int(raw_class_id)

            if class_id >= len(CLASS_NAMES) or class_id < 0:
                continue 
            
            # Since the check is already done by `if confidence >= conf_threshold`, 
            # this section executes for all valid, high-confidence detections.
            
            confs.append(float(confidence))
            class_ids.append(class_id)
            
            # Convert normalized box coordinates (cx, cy, w, h)
            # Box coordinates are still row[0] to row[3]
            cx, cy, w, h = row[:4]
            left = int((cx - w/2) * x_scale)
            top = int((cy - h/2) * y_scale)
            width = int(w * x_scale)
            height = int(h * y_scale)
            
            boxes.append([left, top, width, height])

    # Perform Non-Maximum Suppression (NMS) to eliminate =overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confs, conf_threshold, iou_threshold)
    
    final_detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            final_detections.append({
                'box': boxes[i],
                'conf': confs[i],
                'class_id': class_ids[i]
            })
    return final_detections

# --- MAIN LOOP ---
print(f"Starting webcam inference with {MODEL_PATH}...")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 1. Pre-process the frame
    # Create a 4D blob from the frame (1, 3, 320, 320)
    blob = cv2.dnn.blobFromImage(
        frame, 
        1/255.0, 
        (INPUT_SIZE, INPUT_SIZE), 
        swapRB=True, 
        crop=False
    )

    # 2. Run the model
    raw_output = session.run([output_name], {input_name: blob.astype(np.float32)})
    
    # 3. Post-process the output
    detections = postprocess(raw_output, frame, CONF_THRESHOLD, IOU_THRESHOLD)

    # Insert this line after the postprocess call:
    print(f"Detections found this frame: {len(detections)}")

    # 4. Draw bounding boxes on the frame
    for d in detections:
        box = d['box']
        conf = d['conf']
        class_id = d['class_id']
        label = f"{CLASS_NAMES[class_id]}: {conf:.2f}"
        
        # Draw box and label
        cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the result
    cv2.imshow('YOLOv8 Digit Detection Test', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- CLEANUP ---
cap.release()
cv2.destroyAllWindows()
print("Inference finished. Cleaned up resources.")