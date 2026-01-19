\# Digit Detection Robot



A Raspberry Pi-based autonomous robot that detects handwritten digits (0-4) using YOLOv8n and executes corresponding movement commands in real-time.



!\[Robot Demo](assets/demo.gif)



\## Overview



This project uses computer vision and deep learning to create an interactive robot that responds to digit commands. Show a digit paper to the camera, and the robot executes the corresponding action instantly.



\## Features



\- \*\*Real-time digit detection\*\* using YOLOv8n ONNX model

\- \*\*Instant action execution\*\* with interrupt support

\- \*\*Live web streaming\*\* interface at port 5000

\- \*\*Timed turn actions\*\* for precise control

\- \*\*Ultrasonic obstacle detection\*\* for safety

\- \*\*Continuous movement\*\* for digits 0, 1, 2

\- \*\*Auto-stop turns\*\* for digits 3, 4



\## 🎮 Digit Commands



| Digit | Action | Duration |

|-------|--------|----------|

| \*\*0\*\* | Stop | Instant |

| \*\*1\*\* | Move Forward | Continuous |

| \*\*2\*\* | Move Backward | Continuous |

| \*\*3\*\* | Turn Right | 1.1 seconds |

| \*\*4\*\* | Turn Left | 1.1 seconds |



\*\*Note:\*\* You can interrupt any action by showing a different digit!



\## Hardware Requirements



\### Components

\- Raspberry Pi 4 (4GB RAM recommended) or Raspberry Pi 3B+

\- Robot chassis with 4 DC motors

\- AUPPBot motor driver board

\- USB webcam (minimum 480p resolution)

\- HC-SR04 Ultrasonic distance sensor

\- Servo motor (for camera mount)

\- Power supply (suitable for motors and Pi)



\### Wiring

\- \*\*Ultrasonic Sensor:\*\*

&nbsp; - TRIG → GPIO 21

&nbsp; - ECHO → GPIO 20

\- \*\*Camera:\*\* USB port

\- \*\*Motors:\*\* Connected via AUPPBot controller to `/dev/ttyUSB0`



\## Software Requirements



\- \*\*Operating System:\*\* Raspberry Pi OS (Bullseye or newer)

\- \*\*Python:\*\* 3.7+



\### Dependencies

```bash

opencv-python==4.8.0.74

numpy==1.24.3

onnxruntime==1.15.1

Flask==2.3.2

RPi.GPIO==0.7.1

```



\## Installation



\### 1. Clone the Repository

```bash

git clone https://github.com/Meatra0704/digit-detection-robot.git

cd digit-detection-robot

```



\### 2. Install Dependencies

```bash

pip3 install -r requirements.txt

```



\### 3. Verify Model File

Ensure `best.onnx` is in the `models/` folder:

```bash

ls models/best.onnx

```



\### 4. Configure Hardware

Edit `src/motor\_control.py` if needed:

```python

PORT = "/dev/ttyUSB0"  # Your motor controller port

BAUD = 115200

CAM\_INDEX = 0          # Your camera index

```



\## Usage



\### 1. Run the Robot

```bash

cd src

python3 motor\_control.py

```



\### 2. Access Web Interface

Open a browser and navigate to:

```

http://raspberrypi.local:5000

```

or

```

http://YOUR\_PI\_IP\_ADDRESS:5000

```



\### 3. Control the Robot

\- Hold digit papers (0-4) in front of the camera

\- The robot will detect and execute the corresponding action

\- Show a different digit to interrupt the current action



\## Configuration



\### Adjust Detection Sensitivity

```python

MIN\_DIGIT\_CONFIDENCE = 0.3  # Range: 0.1 (sensitive) to 0.9 (strict)

```



\### Adjust Movement Duration

```python

DIGIT\_ACTION\_DURATION = 1.5  # Duration for forward/backward (seconds)

TURN\_TIME\_90 = 1.1           # Duration for turns (seconds)

```



\### Adjust Motor Speed

```python

BASE = 13      # Base motor speed (range: 0-99)

TURN\_SPEED = 25  # Turn speed (range: 0-99)

```



\### Adjust Cooldown

```python

DIGIT\_COOLDOWN = 0.5  # Seconds between accepting same digit again

```



\## 🏗️ Project Structure

```

digit-detection-robot/

│

├── README.md                    # This file

├── requirements.txt             # Python dependencies

├── .gitignore                   # Git ignore rules

│

├── src/

│   └── motor\_control.py         # Main robot control script

│

├── models/

│   └── best.onnx                # Trained YOLOv8n model

│

├── training/

│   ├── train.py                 # Model training script

│   └── data.yaml                # Dataset configuration

│

└── docs/

&nbsp;   └── setup.md                 # Detailed setup guide

```



\## Training Your Own Model



\### Dataset Preparation

1\. Collect 100+ images per digit (0-4)

2\. Annotate using \[Roboflow](https://roboflow.com/) or LabelImg

3\. Export in YOLOv8 format



\### Training

```bash

\# Install ultralytics

pip install ultralytics



\# Train the model

yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=100 imgsz=320



\# Export to ONNX

yolo export model=runs/detect/train/weights/best.pt format=onnx

```



\### Model Specifications

\- \*\*Architecture:\*\* YOLOv8n (nano)

\- \*\*Input Size:\*\* 320x320 pixels

\- \*\*Classes:\*\* 5 (digits 0, 1, 2, 3, 4)

\- \*\*Format:\*\* ONNX for optimized inference



\## Troubleshooting



\### Robot Not Detecting Digits

\- \*\*Lower confidence threshold:\*\* Set `MIN\_DIGIT\_CONFIDENCE = 0.2`

\- \*\*Improve lighting:\*\* Ensure bright, even lighting on digit papers

\- \*\*Hold closer:\*\* Position digit 30-50cm from camera

\- \*\*Check model:\*\* Verify `best.onnx` exists in models folder



\### Robot Not Moving

\- \*\*Check connections:\*\* Verify motor controller is connected to `/dev/ttyUSB0`

\- \*\*Check power:\*\* Ensure adequate power supply for motors

\- \*\*Test motors:\*\* Run motor self-test on startup

\- \*\*Check permissions:\*\* Run `sudo chmod 666 /dev/ttyUSB0`



\### Camera Issues

\- \*\*Check camera:\*\* Run `ls /dev/video\*` to find camera index

\- \*\*Adjust CAM\_INDEX:\*\* Change in `motor\_control.py`

\- \*\*Test camera:\*\* `python3 -c "import cv2; print(cv2.VideoCapture(0).isOpened())"`



\### Web Interface Not Loading

\- \*\*Check IP address:\*\* Run `hostname -I` on Raspberry Pi

\- \*\*Check firewall:\*\* Ensure port 5000 is open

\- \*\*Try localhost:\*\* Access from Pi directly at `http://localhost:5000`



\### GPU Warning Message

```

\[W:onnxruntime:Default, device\_discovery.cc:164] GPU device discovery failed

```

\*\*This is normal!\*\* Raspberry Pi doesn't have a GPU for deep learning. The model runs on CPU automatically. You can ignore this warning.



\## Performance



\- \*\*Detection Speed:\*\* ~10-15 FPS on Raspberry Pi 4

\- \*\*Detection Accuracy:\*\* 85-95% (depends on training)

\- \*\*Response Time:\*\* <100ms from detection to action

\- \*\*Camera Resolution:\*\* 640x480



\## Advanced Features



\### Interrupt Support

Show digit "1" to move forward, then show "0" mid-movement to stop instantly. All actions can be interrupted by showing a different digit.



\### Timed Turns

Turns (digits 3 and 4) automatically stop after 1.1 seconds, preventing infinite spinning while allowing precise 90° rotations.



\### Web Streaming

Real-time video feed with overlay showing:

\- Current detected digit

\- Confidence percentage

\- Active action status

\- Ultrasonic distance reading



\## 📝 To-Do / Future Improvements



\- \[ ] Add support for more digits (5-9)

\- \[ ] Implement gesture recognition

\- \[ ] Add voice feedback

\- \[ ] Create mobile app controller

\- \[ ] Add autonomous navigation mode

\- \[ ] Support for multiple digit sequences (e.g., "13" = forward then right)



\## Contributing



Contributions are welcome! Please feel free to submit a Pull Request.



1\. Fork the repository

2\. Create your feature branch (`git checkout -b feature/AmazingFeature`)

3\. Commit your changes (`git commit -m 'Add some AmazingFeature'`)

4\. Push to the branch (`git push origin feature/AmazingFeature`)

5\. Open a Pull Request



\## License



This project is licensed under the MIT License - see the \[LICENSE](LICENSE) file for details.



\## Acknowledgments



\- \*\*YOLOv8\*\* by \[Ultralytics](https://github.com/ultralytics/ultralytics)

\- \*\*ONNX Runtime\*\* for optimized inference

\- \*\*OpenCV\*\* for computer vision capabilities

\- \*\*Flask\*\* for web streaming

\- \*\*AUPPBot\*\* library for robot control



\## Authors



\- \*\*Pum Someatra\*\* - \*Initial work\* - \[@meatra0704](https://github.com/meatra0704)



\## Contact



For questions or feedback, please open an issue or contact \[meatra0704@gmail.com](mailto:meatra0704@gmail.com)



---



⭐ If you found this project helpful, please give it a star!

