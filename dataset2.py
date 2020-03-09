import os 
import sys 
import numpy as np 
from PIL import Image 

def _extract_face(filepath, face_cascade): 

    #filepath = '/root/anaconda3/envs/face-recog/yalefaces/subject01.wink' 

    img = Image.open(filepath).convert('L') 
    img = np.array(img, np.uint8) 

    face = face_cascade.detectMultiScale(img) 

    if len(face) != 1:
        sys.exit('Example {} does not have exactly one face'.format(filepath)) 


    face = face[0] 
    x, y, w, h = face 
    face_region = img[y:y+h,x:x+w] 
    return face_region 

def load_data(face_cascade, data_dir='yalefaces'): 


    x_train = [] 
    y_train = [] 

    x_test = [] 
    y_test = [] 

    training_image_files = [f for f in os.listdir(data_dir) if not f.endswith('.wink')] 
    test_image_files = [f for f in os.listdir(data_dir) if f.endswith('.wink')] 

    for image_file in training_image_files: 
        filepath = os.path.join(data_dir, image_file) 

        face_region = extract_face(filepath, face_cascade) 
        person_number = int(image_file.split('.')[0].replace('subject', '') 

        x_train.append(face_region) 
        y_train.append(person_number) 

    for image_file in test_image_files: 
        filepath = os.path.join(data_dir, image_file) 

        face_region = extract_face(filepath, face_cascade) 
        person_number = int(image_file.split('.')[0].replace('subject', '') 

        x_test.append(face_region) 
        y_test.append(person_number) 


    return (x_train, y_train), (x_test, y_test) 
        


            
