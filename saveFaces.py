import subprocess
import sys
import dlib
import cv2
import time
import imutils
import os
from imutils import face_utils

def bash_command(cmd):

    '''Best way to run shell command from python'''

    subprocess.Popen(['/bin/bash', '-c', cmd])


def facialLandMarksRecognition(dirName, name):

    '''In this function the algorithm recognize your
    face and save every frames of the video keeping only the face'''

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    cam = cv2.VideoCapture(0)
    timy = 25
    cont = 0
    sec = (int(round(time.time())))
    deltaSec = (int(round(time.time()))) - sec

    while (deltaSec < timy):

        deltaSec = (int(round(time.time()))) - sec
        val, image = cam.read()
        image = imutils.resize(image, width=700)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)
        title = name+str(cont)+'.jpg'
        dir = os.path.join(dirName, title)

        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(image, (x, y), (x + w + 20, y + h +20), (0, 255, 0), 2)
            img = image[y:y+h+10, x:x+w+10]
            cv2.imwrite(dir, img)
            cont = cont + 1
            cv2.putText(image, "Face #{} {}".format(i + 1, deltaSec), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            for (x, y) in shape:
        	    cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

        cv2.imshow("Output", image)
        cv2.waitKey(1)

    cv2.waitKey(1)
    cv2.destroyAllWindows()

if __name__ == "__main__":

    facialLandMarksRecognition(sys.argv[1], sys.argv[2])
