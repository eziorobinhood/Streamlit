from pickle import FRAME
import cv2
import streamlit as st
import os
from tensorflow import keras
from keras.utils import to_categorical
from keras.layers import Input, Dense 
from keras.models import Model
import numpy as np 
import mediapipe as mp 
from keras.models import load_model 



st.sidebar.title('Underdevelopment - Chakra')
st.title('Pose Detection')
app_mode = st.sidebar.selectbox('Choose app mode',['About app','Data Collection','Training the data','Detection'])

model  = load_model("model.h5")
label = np.load("labels.npy")

holistic = mp.solutions.pose
holis = holistic.Pose()
drawing = mp.solutions.drawing_utils


if app_mode == 'About app':
    st.title('About app')


elif app_mode == 'Data Collection':
    def inFrame(lst):
        if lst[28].visibility > 0.6 and lst[27].visibility > 0.6 and lst[15].visibility>0.6 and lst[16].visibility>0.6:
            return True 
        return False


    cap = cv2.VideoCapture(0)
    name = st.text_input('Enter the name of the asana')
    collect = st.checkbox('Collect data')
    FRAME_WINDOW = st.image([])
    X = []
    data_size = 0
    if collect:
        while True:
            lst = []
            
            ret, frm = cap.read()

            frm = cv2.flip(frm, 1)
            FRAME_WINDOW.image(frm)
            res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

            if res.pose_landmarks and inFrame(res.pose_landmarks.landmark):
                for i in res.pose_landmarks.landmark:
                    lst.append(i.x - res.pose_landmarks.landmark[0].x)
                    lst.append(i.y - res.pose_landmarks.landmark[0].y)

                X.append(lst)
                data_size = data_size+1

            else: 
                cv2.putText(frm, "Make Sure Full body visible", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)

            drawing.draw_landmarks(frm, res.pose_landmarks, holistic.POSE_CONNECTIONS)

            cv2.putText(frm, str(data_size), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)

            
    np.save(f"{name}.npy", np.array(X))
    print(np.array(X).shape)


elif app_mode == 'Training the data':
    is_init = False
    size = -1

    label = []
    dictionary = {}
    c = 0

    for i in os.listdir():
        if i.split(".")[-1] == "npy" and not(i.split(".")[0] == "labels"):  
            if not(is_init):
                is_init = True 
                X = np.load(i)
                size = X.shape[0]
                y = np.array([i.split('.')[0]]*size).reshape(-1,1)
            else:
                X = np.concatenate((X, np.load(i)))
                y = np.concatenate((y, np.array([i.split('.')[0]]*size).reshape(-1,1)))

            label.append(i.split('.')[0])
            dictionary[i.split('.')[0]] = c  
            c = c+1


    for i in range(y.shape[0]):
        y[i, 0] = dictionary[y[i, 0]]
    y = np.array(y, dtype="int32")


    y = to_categorical(y)

    X_new = X.copy()
    y_new = y.copy()
    counter = 0 

    cnt = np.arange(X.shape[0])
    np.random.shuffle(cnt)

    for i in cnt: 
        X_new[counter] = X[i]
        y_new[counter] = y[i]
        counter = counter + 1


    ip = Input(shape=(X.shape[1]))

    m = Dense(128, activation="tanh")(ip)
    m = Dense(64, activation="tanh")(m)

    op = Dense(y.shape[1], activation="softmax")(m) 

    model = Model(inputs=ip, outputs=op) 

    model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])

    model.fit(X_new, y_new, epochs=80)


    model.save("model.h5")
    np.save("labels.npy", np.array(label))


elif app_mode == 'Detection':
    st.header("Pose Detection")
    run = st.checkbox('DETECT')
    def inFrame(lst):
        if lst[28].visibility > 0.6 and lst[27].visibility > 0.6 and lst[15].visibility>0.6 and lst[16].visibility>0.6:
            return True 
        return False

    FRAME_WINDOW = st.image([])

    cam = cv2.VideoCapture(0)

    while run:
        lst=[]
        ret, frame = cam.read()
        cv2.flip(frame,1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)

        window = np.zeros((940,940,3), dtype="uint8")
        res = holis.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.pose_landmarks and inFrame(res.pose_landmarks.landmark):
            for i in res.pose_landmarks.landmark:
                lst.append(i.x - res.pose_landmarks.landmark[0].x)
                lst.append(i.y - res.pose_landmarks.landmark[0].y)
            
            lst = np.array(lst).reshape(1,-1)
            p = model.predict(lst)
            pred = label[np.argmax(p)]

            if p[0][np.argmax(p)] > 0.75:
                cv2.putText(window, pred , (180,180),cv2.FONT_ITALIC, 1.3, (0,255,0),2)
            else:
                cv2.putText(window, "Asana is either wrong not trained" , (100,180),cv2.FONT_ITALIC, 1.8, (0,0,255),3)
        else: 
            cv2.putText(frame, "Make Sure Full body visible", (100,450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255),3)

        drawing.draw_landmarks(frame, res.pose_landmarks, holistic.POSE_CONNECTIONS,
							connection_drawing_spec=drawing.DrawingSpec(color=(255,255,255), thickness=6 ),
							 landmark_drawing_spec=drawing.DrawingSpec(color=(0,0,255), circle_radius=3, thickness=3))