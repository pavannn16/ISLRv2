#venv path ~/PythonVenv/ISLRv2git/bin/python3.12
#Coded by Pavan Chauhan!
import streamlit as st
import cv2
import mediapipe as mp
import pandas as pd
import time
import tensorflow.lite as tflite
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from gtts import gTTS
import os
import requests

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

def create_frame_landmark_df(results, frame, xyz):
    """
    Takes the results from mediapipe and creates a dataframe of the landmarks
    """
    # for having values and rows for every landmark index
    xyz_skel = xyz[['type', 'landmark_index']].drop_duplicates().reset_index(drop=True).copy()
    face = pd.DataFrame()
    pose = pd.DataFrame()
    left_hand = pd.DataFrame()
    right_hand = pd.DataFrame()

    if results.face_landmarks is not None:
        for i, point in enumerate(results.face_landmarks.landmark):
            face.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]
    if results.pose_landmarks is not None:
        for i, point in enumerate(results.pose_landmarks.landmark):
            pose.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]
    if results.left_hand_landmarks is not None:
        for i, point in enumerate(results.left_hand_landmarks.landmark):
            left_hand.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]
    if results.right_hand_landmarks is not None:
        for i, point in enumerate(results.right_hand_landmarks.landmark):
            right_hand.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]

    face = face.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='face')

    pose = pose.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='pose')

    left_hand = left_hand.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='left_hand')

    right_hand = right_hand.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='right_hand')

    landmarks = pd.concat([face, pose, left_hand, right_hand]).reset_index(drop=True)
    # So that skel will have landmarks even if they do not exist
    landmarks = xyz_skel.merge(landmarks, on=['type', 'landmark_index'], how='left')
    # to have actual unique frames
    landmarks = landmarks.assign(frame=frame)
    return landmarks

def capture_video(duration, filename="captured_video.avi"):
    """Captures a video locally for the specified duration."""
    cap = cv2.VideoCapture(0)  # Open the default webcam
    if not cap.isOpened():
        st.error("Error opening webcam. Please check your camera connection.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for saving the video
    out = cv2.VideoWriter(filename, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    start_time = time.time()
    while (time.time() - start_time) < duration:
        ret, frame = cap.read()
        if not ret:
            st.error("Error capturing video frame.")
            break
        out.write(frame)

    cap.release()
    out.release()
    st.success("Video captured successfully!")
    return filename

def process_video(video_path):
    """Processes the captured video file."""
    all_landmarks = []
    frame = 0
    with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
    ) as holistic:
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            landmarks = create_frame_landmark_df(results, frame, xyz)
            all_landmarks.append(landmarks)
            frame += 1

        cap.release()
        return all_landmarks

def load_relevant_data_subset(pq_path):
    ROWS_PER_FRAME = 543  # number of landmarks per frame
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)

def get_prediction(prediction_fn, pq_file):
    xyz_np = load_relevant_data_subset(pq_file)
    prediction = prediction_fn(inputs=xyz_np)
    pred = prediction['outputs'].argmax()
    sign_ord = pred.item()
    sign = ORD2SIGN[sign_ord]
    pred_conf = prediction['outputs'][pred]
    st.write(f'PREDICTED SIGN: {sign} [{sign_ord}], CONFIDENCE: {pred_conf:0.4}')
    
    # Convert text to speech using gTTS
    tts = gTTS(text=f'The predicted sign is {sign}', lang='en')
    tts.save("predicted_sign.mp3")
    os.system("mpg321 predicted_sign.mp3")


def animate_sign_video(sign):
    def get_hand_points(hand):
        x = [[hand.iloc[0].x, hand.iloc[1].x, hand.iloc[2].x, hand.iloc[3].x, hand.iloc[4].x], # Thumb
            [hand.iloc[5].x, hand.iloc[6].x, hand.iloc[7].x, hand.iloc[8].x], # Index
            [hand.iloc[9].x, hand.iloc[10].x, hand.iloc[11].x, hand.iloc[12].x], 
            [hand.iloc[13].x, hand.iloc[14].x, hand.iloc[15].x, hand.iloc[16].x], 
            [hand.iloc[17].x, hand.iloc[18].x, hand.iloc[19].x, hand.iloc[20].x], 
            [hand.iloc[0].x, hand.iloc[5].x, hand.iloc[9].x, hand.iloc[13].x, hand.iloc[17].x, hand.iloc[0].x]]

        y = [[hand.iloc[0].y, hand.iloc[1].y, hand.iloc[2].y, hand.iloc[3].y, hand.iloc[4].y],  #Thumb
            [hand.iloc[5].y, hand.iloc[6].y, hand.iloc[7].y, hand.iloc[8].y], # Index
            [hand.iloc[9].y, hand.iloc[10].y, hand.iloc[11].y, hand.iloc[12].y], 
            [hand.iloc[13].y, hand.iloc[14].y, hand.iloc[15].y, hand.iloc[16].y], 
            [hand.iloc[17].y, hand.iloc[18].y, hand.iloc[19].y, hand.iloc[20].y], 
            [hand.iloc[0].y, hand.iloc[5].y, hand.iloc[9].y, hand.iloc[13].y, hand.iloc[17].y, hand.iloc[0].y]] 
        return x, y

    def get_pose_points(pose):
        x = [[pose.iloc[8].x, pose.iloc[6].x, pose.iloc[5].x, pose.iloc[4].x, pose.iloc[0].x, pose.iloc[1].x, pose.iloc[2].x, pose.iloc[3].x, pose.iloc[7].x], 
            [pose.iloc[10].x, pose.iloc[9].x], 
            [pose.iloc[22].x, pose.iloc[16].x, pose.iloc[20].x, pose.iloc[18].x, pose.iloc[16].x, pose.iloc[14].x, pose.iloc[12].x, 
            pose.iloc[11].x, pose.iloc[13].x, pose.iloc[15].x, pose.iloc[17].x, pose.iloc[19].x, pose.iloc[15].x, pose.iloc[21].x], 
            [pose.iloc[12].x, pose.iloc[24].x, pose.iloc[26].x, pose.iloc[28].x, pose.iloc[30].x, pose.iloc[32].x, pose.iloc[28].x], 
            [pose.iloc[11].x, pose.iloc[23].x, pose.iloc[25].x, pose.iloc[27].x, pose.iloc[29].x, pose.iloc[31].x, pose.iloc[27].x], 
            [pose.iloc[24].x, pose.iloc[23].x]
            ]

        y = [[pose.iloc[8].y, pose.iloc[6].y, pose.iloc[5].y, pose.iloc[4].y, pose.iloc[0].y, pose.iloc[1].y, pose.iloc[2].y, pose.iloc[3].y, pose.iloc[7].y], 
            [pose.iloc[10].y, pose.iloc[9].y], 
            [pose.iloc[22].y, pose.iloc[16].y, pose.iloc[20].y, pose.iloc[18].y, pose.iloc[16].y, pose.iloc[14].y, pose.iloc[12].y, 
            pose.iloc[11].y, pose.iloc[13].y, pose.iloc[15].y, pose.iloc[17].y, pose.iloc[19].y, pose.iloc[15].y, pose.iloc[21].y], 
            [pose.iloc[12].y, pose.iloc[24].y, pose.iloc[26].y, pose.iloc[28].y, pose.iloc[30].y, pose.iloc[32].y, pose.iloc[28].y], 
            [pose.iloc[11].y, pose.iloc[23].y, pose.iloc[25].y, pose.iloc[27].y, pose.iloc[29].y, pose.iloc[31].y, pose.iloc[27].y], 
            [pose.iloc[24].y, pose.iloc[23].y]
            ]
        return x, y

    def animation_frame(f):
        frame = sign[sign.frame==f]
        left = frame[frame.type=='left_hand']
        right = frame[frame.type=='right_hand']
        pose = frame[frame.type=='pose']
        face = frame[frame.type=='face'][['x', 'y']].values
        lx, ly = get_hand_points(left)
        rx, ry = get_hand_points(right)
        px, py = get_pose_points(pose)
        ax.clear()
        ax.plot(face[:,0], face[:,1], '.')
        for i in range(len(lx)):
            ax.plot(lx[i], ly[i])
        for i in range(len(rx)):
            ax.plot(rx[i], ry[i])
        for i in range(len(px)):
            ax.plot(px[i], py[i])
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)

    
    # These values set the limits on the graph to stabilize the video
    xmin = sign.x.min() - 0.2
    xmax = sign.x.max() + 0.2
    ymin = sign.y.min() - 0.2
    ymax = sign.y.max() + 0.2
    fig, ax = plt.subplots()
    l, = ax.plot([], [])
    animation = FuncAnimation(fig, func=animation_frame, frames=sign.frame.unique())

    return animation.to_html5_video()

def download_file(url):
    """Downloads a file from a given URL and saves it locally."""
    local_filename = url.split('/')[-1]  # Get filename from URL
    with requests.get(url, stream=True) as r:
        r.raise_for_status()  # Raise an exception if there's an error
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename


if __name__ == "__main__":
    #Raw GitHub URLs for your files (Make sure to update with your repo)
    dummy_parquet_skel_file_url = 'https://raw.githubusercontent.com/pavannn16/ISLRv2/main/data/239181.parquet'
    tflite_model_url = 'https://raw.githubusercontent.com/pavannn16/ISLRv2/main/models/asl_model.tflite'
    csv_file_url ='https://raw.githubusercontent.com/pavannn16/ISLRv2/main/data/train.csv'
    captured_parquet_file = "shammers.parquet"  # Keep this local

    # Download the necessary files (if they don't exist locally)
    if not os.path.exists("239181.parquet"):
        download_file(dummy_parquet_skel_file_url)
    if not os.path.exists("asl_model.tflite"):
        download_file(tflite_model_url)
    if not os.path.exists("train.csv"):
        download_file(csv_file_url)

    xyz = pd.read_parquet("239181.parquet")

    # Combine main script and inference code
    interpreter = tflite.Interpreter(model_path="asl_model.tflite")
    found_signatures = list(interpreter.get_signature_list().keys())
    prediction_fn = interpreter.get_signature_runner("serving_default")
    train = pd.read_csv("train.csv")
    # Add ordinally Encoded Sign (assign number to each sign name)
    train['sign_ord'] = train['sign'].astype('category').cat.codes

    # Dictionaries to translate sign <-> ordinal encoded sign
    SIGN2ORD = train[['sign', 'sign_ord']].set_index('sign').squeeze().to_dict()
    ORD2SIGN = train[['sign_ord', 'sign']].set_index('sign_ord').squeeze().to_dict()

    # Streamlit app layout
    st.title("Isolated Sign Language Recognition App")
    st.write("Set the duration (in seconds) and press the 'Predict Sign' button to capture your sign and get the prediction along with the animated visuals of the captured landmarks.")
    duration = st.number_input("Set Duration (in seconds)", min_value=1)
    if st.button("Predict Sign") and duration:
        captured_video_path = capture_video(duration)
        if captured_video_path:
            captured_landmarks = process_video(captured_video_path)
            if captured_landmarks:
                captured_landmarks_df = pd.concat(captured_landmarks).reset_index(drop=True)
                captured_landmarks_df.to_parquet(captured_parquet_file)
                sign = pd.read_parquet(captured_parquet_file)
                sign.y = sign.y * -1

                # Display animated video
                st.write(animate_sign_video(sign), unsafe_allow_html=True)

                # Make prediction
                get_prediction(prediction_fn, captured_parquet_file) 