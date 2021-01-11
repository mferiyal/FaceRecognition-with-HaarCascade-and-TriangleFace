from math import sqrt
import cv2
import sys
import numpy
import pandas
import RPi.GPIO as GPIO
import time
from datetime import datetime
from pytz import timezone
from PyQt5.QtCore import QThread

flagBuffer = False

fmt = '%Y-%m-%d %H:%M:%S'
# define eastern timezone
eastern = timezone('Asia/Jakarta')

GPIO.setmode(GPIO.BCM)
GPIO.setup(18, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(16, GPIO.OUT)
GPIO.setup(21, GPIO.OUT)
GPIO.setup(17, GPIO.OUT)

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
video_capture = cv2.VideoCapture(0)
countingEye = 0
x=y=w=h=0
centerOfEye = [[0,0], [0,0]]
saveCenterEye = [[0,0], [0,0]]
rectEye = [[0,0], [0,0]]
saveRectEye = [[0,0], [0,0]]
centerOfMouth = [0,0]
centerOfNose = [0,0]
faceFeature = numpy.array([0,0,0,0,0])
faceNormalization = numpy.array([0,0,0,0,0])
faceFeature = faceFeature.astype(float)
faceNormalization = faceNormalization.astype(float)
resultData = numpy.array([1000,1000,1000,1000,1000,1000,1000,1000])
resultData = resultData.astype(float)
copyResultData = numpy.copy(resultData)
countingFace = 0
lastCountingFace = 0
flagCountingFace = 0
def normalization(featureInput):
    for valueFeat in range(len(featureInput)):
        faceNormalization[valueFeat] = featureInput[valueFeat] / numpy.sum(featureInput)
while True:
    # Capture frame-by-frame
    retval, frame = video_capture.read()
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect features specified in Haar Cascade
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(35, 35)
    )
    if x == 0 and y == 0 and w == 0 and h == 0:
        flagBuffer = False

    if countingFace == lastCountingFace:
	flagCountingFace += 1
    else:
	flagCountingFace = 0

    lastCountingFace = countingFace

    if flagCountingFace > 50:
	GPIO.output(17, GPIO.LOW)
	lastCountingFace = countingFace = flagCountingFace = 0
	time.sleep(3)

    #print("xyh", x, y, w, h)
    # Draw a rectangle around recognized faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (200, 200, 200), 2)
	countingFace += 1
	flagCountingFace = 0
	GPIO.output(17, GPIO.HIGH)

        # Detect features specified in Haar Cascade
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            centerOfEyeX = int(ex + (ew/2)) + x
            centerOfEyeY = int(ey + (eh/2)) + y
            rectEye[countingEye][0] = ex + x
            rectEye[countingEye][1] = ew
            lineCenterW = int(x + (w/2))
            lineCenterH = int(y + (h/2)) * 0.9
            if centerOfEyeY < lineCenterH:
                if centerOfEyeX < lineCenterW:
                    saveCenterEye[0][0] = centerOfEyeX
                    saveCenterEye[0][1] = centerOfEyeY
                    cv2.circle(frame,(centerOfEyeX, centerOfEyeY), 5, (0,255,0), -1)
                else:
                    saveCenterEye[1][0] = centerOfEyeX
                    saveCenterEye[1][1] = centerOfEyeY
                    cv2.circle(frame,(centerOfEyeX, centerOfEyeY), 5, (255,255,255), -1)

                if saveCenterEye[0][0] != 0 and saveCenterEye[1][0] != 0:
                    faceFeature[0] = int(sqrt(((saveCenterEye[countingEye][0] - saveCenterEye[countingEye-1][0]) ** 2) + ((saveCenterEye[countingEye][1] - saveCenterEye[countingEye][1]) ** 2)))
                    saveCenterEye[0][0] = saveCenterEye[0][1] = saveCenterEye[1][0] = saveCenterEye[1][1] = 0


        mouth = mouth_cascade.detectMultiScale(roi_gray)
        for (mx, my, mw, mh) in mouth:
            maxH = int(y + (0.96 * h))
            minH = int(y + (0.75 * h))
            centerOfMouth[0] = int(mx + (mw/2)) + x
            centerOfMouth[1] = int(my + (mh/2)) + y
            if centerOfMouth[1] > minH and centerOfMouth[1] <= maxH:
                #cv2.rectangle(frame,(x, minH), (x+w, maxH), (0,0,255), 1) #min max mouth
                faceFeature[1] = int(sqrt(((saveCenterEye[0][0] - centerOfMouth[0]) ** 2) + ((saveCenterEye[0][1] - centerOfMouth[1]) ** 2)))
                faceFeature[2] = int(sqrt(((saveCenterEye[1][0] - centerOfMouth[0]) ** 2) + ((saveCenterEye[1][1] - centerOfMouth[1]) ** 2)))
                cv2.circle(frame,(centerOfMouth[0],centerOfMouth[1]), 5, (0,0,255), -1)
                cv2.rectangle(roi_color,(mx,my),(mx+mw,my+mh),(0,0,255),2)

        nose = nose_cascade.detectMultiScale(roi_gray)
        for (nx, ny, nw, nh) in nose:
            minX = int(x + (w/3))
            maxX = int((x+w) - (w/3))
            minY = int(y + (h/3))
            maxY = int((y+h) - (h/3))
            centerOfNose[0] = int(nx + (nw/2)) + x
            centerOfNose[1] = int(ny + (nh/2)) + y
            if centerOfNose[0] > minX and centerOfNose[0] <= maxX and centerOfNose[1] > minY and centerOfNose[1] <= maxY:
                faceFeature[3] = int(sqrt(((saveCenterEye[0][0] - centerOfNose[0]) ** 2) + ((saveCenterEye[0][1] - centerOfNose[1]) ** 2)))
                faceFeature[4] = int(sqrt(((saveCenterEye[1][0] - centerOfNose[0]) ** 2) +((saveCenterEye[1][1] - centerOfNose[1]) ** 2) ))
                cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(255,0,0),2)
                cv2.circle(frame,(centerOfNose[0],centerOfNose[1]), 5, (255,0,0), -1)
                #cv2.rectangle(frame,(minX,minY),(maxX,maxY),(255,255,255),1) #1/3 kotak putih

        normalization(faceFeature)
   	df = pandas.read_csv('dataseet.csv')
    	heigh = df.shape[0]
    	width = df.shape[1]
	#print("ukuran", heigh, width)
    	for ii in range(heigh):
        	for jj in range(1,width-1):
			#print("ke", ii, jj)
            		resultData[ii] += (df.iloc[ii, jj] - faceFeature[jj-1]) ** 2
            	resultData[ii] = numpy.sqrt(resultData[ii])
		print("data", ii, faceFeature, resultData[ii])
	copyResultData = numpy.copy(resultData)
	resultData.sort()
	if resultData[0] < 25 and flagBuffer == False:
		for i in range(0, len(copyResultData)-1):
			if resultData[0] == copyResultData[i]:
				loc_dt = datetime.now(eastern)
				timing = loc_dt.strftime(fmt)
                                if i < heigh:
				#print(df.iloc[3,6])
					dataLogging = pandas.DataFrame([[df.iloc[i, width-1],timing]],columns=['nama','waktu'])
					dataLogging.to_csv('datalog.csv', mode='a', header=False)
					#print("oke")
                flagBuffer = True
                #print("cocok")
		GPIO.output(16, GPIO.HIGH)
                GPIO.output(21, GPIO.HIGH)
		time.sleep(5)
		GPIO.output(16,  GPIO.LOW)
                GPIO.output(21, GPIO.LOW)
                x=y=w=h=0
                print("sleep")
                QThread.sleep(1)
                print("wakeup")
                video_capture = None
                video_capture = cv2.VideoCapture(0)
		GPIO.output(17, GPIO.LOW)

    	btnTraining = GPIO.input(18)
    	if btnTraining == False:
		dataTrain = pandas.DataFrame([[faceFeature[0],
                                            faceFeature[1],
					    faceFeature[2],
					    faceFeature[3],
					    faceFeature[4],
                                            "user"]],
                                          columns=['1', '2','3','4','5','nama'])
		dataTrain.to_csv('dataseet.csv', mode='a', header=False)
		print("data saved")
		GPIO.output(17, GPIO.LOW)

    # Display the resulting frame
    cv2.imshow('Video', frame)
    # Exit the camera view
    if cv2.waitKey(1) & 0xFF == ord('q'):
        sys.exit()
	GPIO.cleanup()
