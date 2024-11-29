import time
import cv2
import threading
from flask import Flask, render_template, Response
import HandTrackingModule as htm  # Your hand tracking module

app = Flask(__name__)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize the hand detector
detector = htm.handDetector(detectionCon=0.65, maxHands=1)

# Function to generate frames for the webcam
def gen_frames():
    while True:
        success, img = cap.read()
        if not success:
            break
        else:
            # Find hands and get their positions
            img = detector.findHands(img)
            lmList = detector.findPosition(img, draw=False)

            # Example of using counting fingers functionality
            if len(lmList) != 0:
                fingers = detector.fingersUp()  # Get which fingers are up
                totalFingers = fingers.count(1)
                print(f"Total fingers up: {totalFingers}")

            # Convert the image to JPEG and yield it as a frame for the HTML page
            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    # Render the main page with a selection for painting or counting fingers
    return render_template('index.html')

@app.route('/painting')
def painting():
    # Render the page for virtual painting
    return render_template('painting.html')

@app.route('/counting')
def counting():
    # Render the page for finger counting
    return render_template('counting.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/test')
def test():
    return render_template('index.html')


if __name__ == "__main__":
    # Start the Flask app
    app.run(debug=True, threaded=True)
