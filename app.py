import time
import cv2
import threading
from flask import Flask, render_template, Response
import os  # For handling directories and file paths
import numpy as np  # For array operations (used in VirtualPainter)
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

def gen_frames_painting():
    folderPath = "Header"
    myList = os.listdir(folderPath)
    overlayList = [cv2.imread(f'{folderPath}/{imPath}') for imPath in myList]

    header = overlayList[0]  # Default header
    drawColor = (255, 0, 255)
    brushThickness = 25
    eraserThickness = 100

    xp, yp = 0, 0

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)  # Set width
    cap.set(4, 720)   # Set height

    success, img = cap.read()
    if not success:
        return

    imgCanvas = np.zeros_like(img)  # Initialize canvas with same shape as frame

    while True:
        success, img = cap.read()
        if not success:
            break
        img = cv2.flip(img, 1)  # Flip horizontally

        # 1. Find hand landmarks
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)

        if len(lmList) != 0:
            x1, y1 = lmList[8][1:]  # Index finger tip
            x2, y2 = lmList[12][1:]  # Middle finger tip

            # Check which fingers are up
            fingers = detector.fingersUp()

            # Selection mode: Two fingers up
            if fingers[1] and fingers[2]:
                xp, yp = 0, 0
                if y1 < 125:
                    if 250 < x1 < 450:
                        header = overlayList[0]
                        drawColor = (255, 0, 255)
                    elif 550 < x1 < 750:
                        header = overlayList[1]
                        drawColor = (255, 0, 0)
                    elif 800 < x1 < 950:
                        header = overlayList[2]
                        drawColor = (0, 255, 0)
                    elif 1050 < x1 < 1200:
                        header = overlayList[3]
                        drawColor = (0, 0, 0)
                cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

            # Drawing mode: Index finger up
            if fingers[1] and not fingers[2]:
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1
                if drawColor == (0, 0, 0):
                    cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
                else:
                    cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
                xp, yp = x1, y1

        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, imgInv)
        img = cv2.bitwise_or(img, imgCanvas)

        # Setting the header image
        header_resized = cv2.resize(header, (img.shape[1], 125))  # Resize header to match frame width
        img[0:125, 0:img.shape[1]] = header_resized

        # Encode image to bytes
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
@app.route('/finger_count')
def finger_count():
    # Example of returning the number of fingers detected (update logic as needed)
    if detector:
        fingers = detector.fingersUp()
        totalFingers = fingers.count(1)
        return {"count": totalFingers}
    return {"count": 0}
@app.route('/video_feed_painting')
def video_feed_painting():
    return Response(gen_frames_painting(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    # Start the Flask app
    app.run(debug=True, threaded=True)
