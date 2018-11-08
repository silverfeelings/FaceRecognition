import os
import cv2
import numpy as np
from PIL import Image
import pickle

faceClassifier = cv2.CascadeClassifier('classifierModel.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "people")
#print(image_dir)

curr_ids = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("jpg"):
            path = os.path.join(root, file)
            #print(path)
            label = os.path.basename(root).lower()
            if not label in label_ids:
                label_ids[label] = curr_ids
                curr_ids += 1
            id_ = label_ids[label]
            y_labels.append(id_)
            pil_image = Image.open(path).convert("L")
            image_array = np.array(pil_image, "uint8")
            x_train.append(image_array)
            

with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainer.yml")
print("Dataset generated successfully")
