import os
import pickle
import math
import cv2
from subRoutines import *

faceClassifier = FaceDetector("classifierModel.xml")
webcam = VideoCamera()

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items() }

while True:
        frame = webcam.get_frame()
        face_cood = faceClassifier.detect(frame, True)
        if len(face_cood):
            faces = normalize_images(frame, face_cood)
            for i, face in enumerate(faces):
                id_, conf = recognizer.predict(face)
                threshold = 138
                name = labels[id_].capitalize()
                cood = (face_cood[i][0], face_cood[i][1] - 5)
                if conf < threshold:
                        cv2.putText(frame, name, cood, font, 3, color, 2)
                else:
                        cv2.putText(frame, "Unknown", cood, font, 3, color, 2)
                draw_rectangle(frame, face_cood)
        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(20) & 0xff == 27:
            cv2.destroyAllWindows()
            del webcam
            break
