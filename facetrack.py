from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from scipy import misc
import cv2
import numpy as np
import facenet
import time
from datetime import datetime
import detect_face
import os
import time
import imutils
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import pickle
import json
from json import JSONEncoder

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


input_video="akshay_mov.mp4"
modeldir = './model/20170511-185253.pb'
classifier_filename = './class/classifier.pkl'
npy='./npy'
train_img="./train_img"
margin = 44

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def to_json(ID, emb_array):

    fname = 'embedding.json'
    emb_array = np.array(emb_array)
    
    # Data to be written 
    dictionary ={ 
        ID : emb_array 
    } 
    
      
    # Writing to sample.json 
    
    if not os.path.isfile(fname):
        data = dictionary

    else:
        data = json.load(open(fname))
        data = json.loads(data)
        data.update(dictionary)
    
    data = json.dumps(data, cls=NumpyArrayEncoder)
    with open(fname, 'w') as outfile:
            json.dump(data, outfile, indent = 4)

def from_json():
    f = open('embedding.json',)
    data = json.load(f)
    data = json.loads(data)
    return data


with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)

        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        margin = 44
        frame_interval = 1
        batch_size = 1000
        image_size = 182
        input_image_size = 160
        
        print('Loading Model')
        facenet.load_model('model.pb')
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]


        classifier_filename_exp = os.path.expanduser('my_classifier.pkl')
        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile)

        vs = WebcamVideoStream(src=0).start()
        fps = FPS().start()
        c = 0


        print('Start Recognition')
        prevTime = 0
        embedding = []
        count = 0
        ID = []
        while True:
            frame = vs.read()

            frame = cv2.resize(frame, (0,0), fx=0.7, fy=0.7)    #resize frame (optional)
            #dateTimeObj = datetime.now()
            #timestampStr = dateTimeObj.strftime("(%H:%M:%S.%f)")

            curTime = time.time()+1    # calc fps
            timeF = frame_interval

            if (c % timeF == 0):
                find_results = []
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if frame.ndim == 2:
                    frame = facenet.to_rgb(frame)
                frame = frame[:, :, 0:3]
                bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                nrof_faces = bounding_boxes.shape[0]
                print('Detected_FaceNum: %d' % nrof_faces)

                if nrof_faces > 0:
                    det = bounding_boxes[:, 0:4]
                    img_size = np.asarray(frame.shape)[0:2]

                    cropped = []
                    scaled = []
                    scaled_reshape = []

                    bb = np.zeros((nrof_faces,4), dtype=np.int32)
                    for i in range(nrof_faces):
                        present = False
                        emb_array = np.zeros((1, embedding_size))

                        bb[i][0] = np.maximum(det[i][0]-margin/2, 0)
                        bb[i][1] = np.maximum(det[i][1]-margin/2, 0)
                        bb[i][2] = np.minimum(det[i][2]+margin/2, img_size[1])
                        bb[i][3] = np.minimum(det[i][3]+margin/2, img_size[0])

                        cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                        cropped[i] = facenet.flip(cropped[i], False)
                        scaled.append(misc.imresize(cropped[i], (image_size, image_size), interp='bilinear'))
                        scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),
                                               interpolation=cv2.INTER_CUBIC)
                        scaled[i] = facenet.prewhiten(scaled[i])
                        scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))
                        feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                        emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                        if (count == 0):
                    
                            #embedding.append(emb_array)
                            #ID.append(i)
                            #real_ID = ID[i]
                            real_ID = i
                            to_json(i,emb_array)

                        else:
                            data = from_json()
                            for k in range(0, len(data)):
                                dist = np.sqrt(np.sum(np.square(np.subtract(emb_array, list(data.values())[k]))))
                                if(dist<1.0):
                                    to_json(k,emb_array)
                                    #embedding[k] = emb_array
                                    #real_ID = ID[k]
                                    real_ID = k
                                    present = True
                                    break

                            if (present == False):
                                index = int(len(data) + 1)
                                to_json(index,emb_array)
                                #embedding.append(emb_array)
                                #ID.append(int(len(ID) + 1))
                                #real_ID = int(len(ID) + 1)
                                real_ID = index

                        predictions = model.predict_proba(emb_array)
                        best_class_indices = np.argmax(predictions, axis=1)
                        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

                        if best_class_probabilities[0] > 0.5:
                            name = class_names[best_class_indices[0]]
                        else:
                            name = "Unknown"
                        # print("predictions")
                        print(best_class_indices,' with accuracy ',best_class_probabilities)

                        # print(best_class_probabilities)
                        cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)    #boxing face

                            #plot result idx under box
                        text_x = bb[i][0]
                        text_y = bb[i][3] + 20
                            
                        cv2.putText(frame, name + " ID  " + str(real_ID), (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                    1, (0, 0, 255), thickness=1, lineType=2)
                else:
                    print('Alignment Failure')

            #print(str(c) + "  " + str(timestampStr))
            #c = c+1
            count = count +1
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()
