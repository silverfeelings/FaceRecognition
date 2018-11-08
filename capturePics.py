import numpy
import os
import math
import cv2
from IPython.display import clear_output
from matplotlib import pyplot as plt
from subRoutines import *

webcam = VideoCamera()
detector = FaceDetector("classifierModel.xml")

folder = 'people/' + input('Person: ').lower()

if not os.path.exists(folder):
    os.mkdir(folder)
    cnt = 0
    tmr = 0
    while cnt < 10:
        frame = webcam.get_frame()
        face_cood = detector.detect(frame)
        if len(face_cood) and (tmr % 500) == 50:
            faces = cut_faces(frame, face_cood)
            faces = normalize_intensity(faces)
            faces = resize(faces)
            cv2.imwrite(folder + '/' + str(cnt) + '.jpg', faces[0])
            plt_imshow(faces[0], "Images saved: " + str(cnt) )
            clear_output(wait = True)
            cnt += 1
        cv2.waitKey(20)
        tmr += 50
    cv2.destroyAllWindows()
else:
    print("Already exists")
