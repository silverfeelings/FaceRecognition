import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

font = cv2.FONT_HERSHEY_PLAIN
color = (0,0,250)
stroke = 2

#---------------------------------------------------------------------------#
def plt_imshow(image, title=''):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.axis('off')
    plt.title(title)
    plt.imshow(image, cmap='Greys_r')
    plt.show()

def draw_rectangle(frame, face_cood):
    for (x,y,w,h) in face_cood:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (250,30,30), 2)

#---------------------------------------------------------------------------#
def cut_faces(image, face_cood):
    faces = []
    for (x,y,w,h) in face_cood:
        w_rm = int(0.2 * w/2)
        faces.append(image[y:y+h, x+w_rm:x+w-w_rm])
    return faces

#---------------------------------------------------------------------------#
def normalize_intensity(images):
    images_norm = []
    for image in images:
        is_color = len(image.shape) == 3
        if is_color:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        images_norm.append(cv2.equalizeHist(image))
    return images_norm

#---------------------------------------------------------------------------#
def resize(images, size=(50,50)):
    images_norm = []
    for image in images:
        if image.shape < size:
            image_norm = cv2.resize(image, size, interpolation = cv2.INTER_AREA)
        else:
            image_norm = cv2.resize(image, size, interpolation = cv2.INTER_CUBIC)
        images_norm.append(image_norm)
    return images_norm

#---------------------------------------------------------------------------#
def normalize_images(frame, face_cood):
    faces = cut_faces(frame, face_cood)
    faces = normalize_intensity(faces)
    faces = resize(faces)
    return faces

#---------------------------------------------------------------------------#
class VideoCamera():
    def __init__(self, index=0):
        self.video = cv2.VideoCapture(index)
        self.index = index
        print(self.video.isOpened())
    def __del__(self):
        self.video.release()
    def get_frame(self, in_grayscale=False):
        ret, frame = self.video.read()
        if in_grayscale:
            frame = cv2.cvtColor(frame, COLOR_BGR2GRAY)
        return frame

#---------------------------------------------------------------------------#
class FaceDetector():
    def __init__(self, xml_path):
        self.classifier = cv2.CascadeClassifier(xml_path)
    def detect(self, image, biggest_only=True):
        scale_factor = 1.2
        min_neighbor = 5
        min_size = (30,30)
        biggest_only = True
        flags = cv2.CASCADE_FIND_BIGGEST_OBJECT |\
                cv2.CASCADE_DO_ROUGH_SEARCH if biggest_only else \
                cv2.CASCADE_SCALE_IMAGE
        face_cood = self.classifier.detectMultiScale(image,
                                                     scaleFactor=scale_factor,
                                                     minNeighbors=min_neighbor,
                                                     minSize=min_size,
                                                     flags=flags )
        return face_cood


