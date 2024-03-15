import cv2
import numpy as np
from pygame import mixer
from tensorflow.keras.models import load_model


from flask import Flask, redirect, request, render_template

app=Flask(__name__)

@app.route('/' , methods=['GET', 'POST'])
def start():
    return render_template('frontend.html')

@app.route('/live', methods=['GET', 'POST'])
def activate():
    
    mixer.init()
    sound = mixer.Sound('./alarm.wav')
    cap = cv2.VideoCapture(0)
    Score = 0

    # Load pre-trained face detector
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Load pre-trained eye detector
    eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # Load your pre-trained model
    model = load_model('./models/model.h5')

    frame_skip = 2
    frame_count = 0

    while True:
        ret, frame = cap.read()
        frame_count += 1

        if frame_count % frame_skip != 0:
            continue

        height, width = frame.shape[0:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Increase brightness
        frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=20)

        # Use pre-trained face detector
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        cv2.rectangle(frame, (0, height-50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Draw rectangle around the face

            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_detector.detectMultiScale(roi_gray)

            for (ex, ey, ew, eh) in eyes:
                eye = frame[y+ey:y+ey+eh, x+ex:x+ex+ew]
                eye = cv2.resize(eye, (80, 80))
                eye = eye / 255.0
                eye = np.expand_dims(eye, axis=0)

                # Model prediction
                prediction = model.predict(eye)

                if prediction[0][1] > 0.6:  # Adjust the threshold based on your model's characteristics
                    cv2.putText(frame, 'open', (10, height-20), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                fontScale=1, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
                    Score = max(0, Score - 1)
                else:
                    cv2.putText(frame, 'closed', (10, height-20), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                fontScale=1, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
                    Score += 1
                    if Score > 5 :  # Only play the sound if at least one eye is detected
                        try:
                            sound.play()
                        except:
                            pass

                cv2.putText(frame, 'Score' + str(Score), (100, height-20), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            fontScale=1, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

        cv2.imshow('frame', frame)

        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return render_template('frontend.html')

    
    
    
    
@app.route('/image', methods=['GET', 'POST'])
def detimg():
    if(request.method=="POST"):
        imgpath = request.form['path']
    
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    model = load_model('./models/model.h5') 
    frame = cv2.imread(imgpath)
    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=20)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    c=0

    for (ex, ey, ew, eh) in eyes:
        eye = frame[ey:ey + eh, ex:ex + ew]
        eye = cv2.resize(eye, (80, 80))
        eye = eye / 255.0
        eye = np.expand_dims(eye, axis=0)

        # Make predictions using the eye model
        prediction = model.predict(eye)

        # Check the prediction
        if prediction[0][1] > 0.6:
            return render_template('camresult.html',result='Open')
        break
    return render_template('camresult.html',result='Closed')
    
    
    
    
    
@app.route('/video' , methods=['GET','POST'])
def detvideo():
    if(request.method=="POST"):
        vidpath = request.form['path']
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # Load the pre-trained eye detection model
    model = load_model('./models/model.h5')

    # Open the video file
    cap = cv2.VideoCapture(vidpath)

    # Set the frame skip parameters
    frame_skip = 10
    frame_count = 0
    res={'Open':0,'Closed':0}
    vidres=[]
    score=0
    while True:
        ret, frame = cap.read()
        frame_count += 1

        if frame_count % frame_skip != 0:
            continue

        if not ret:
            break  # Break the loop if the video ends

        # Increase brightness
        frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=20)

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect eyes in the frame
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        for (ex, ey, ew, eh) in eyes:
            eye = frame[ey:ey + eh, ex:ex + ew]
            eye = cv2.resize(eye, (80, 80))
            eye = eye / 255.0
            eye = np.expand_dims(eye, axis=0)

            # Make predictions using the eye model
            prediction = model.predict(eye)

            # Check the prediction and print the result
            if prediction[0][1] > 0.6:
                status = 'Open'
                score=0
            else:
                status = 'Closed'
                score+=1
                if(score>3):
                    return render_template('vidresult.html',result='Closed',flag=0,status={cap.get(cv2.CAP_PROP_POS_MSEC) / 1000})
            res[status]+=1
            print(f"Eye status at {cap.get(cv2.CAP_PROP_POS_MSEC) / 1000:.1f} seconds: {status}")
            vidres.append(f"Eye status at {cap.get(cv2.CAP_PROP_POS_MSEC) / 1000:.1f} seconds: {status}")
    print(res['Closed'],res['Open'])
    if(res['Closed']<(res['Open'])):
        return render_template('vidresult.html',result='Open',flag=1)
    else:
        return render_template('vidresult.html',result='Closed',flag=0,status={cap.get(cv2.CAP_PROP_POS_MSEC) / 1000})
    
    
    
    
    
if __name__=="__main__" :
    app.run()
