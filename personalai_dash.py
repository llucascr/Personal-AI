import streamlit as st
import pandas as pd 
from personal_ai import *

st.set_page_config(layout="wide")

personalAI = PersonalAI("IMG_3429.mp4")
personalAI.run()

placeholder = st.empty()

status = "relaxed"
cont = 0

while True:
    frame, results, ts = personalAI.image_q.get()
    if ts == "done":
        break

    if len(results.pose_landmarks) > 0:
        elbow_angle = personalAI.find_angle(results, 12, 14, 16)
        hip_angle = personalAI.find_angle(results, 11, 23, 25)

        # Lógica de apoio / Flexões
        if elbow_angle > 150 and hip_angle > 170:
            status = "ready"
            direction = "down"
        
        if status == "ready":
            if direction == "down" and elbow_angle < 50:
                direction = "up"
                cont += 0.5
            elif direction == "up" and elbow_angle > 100:
                direction = "down"
                cont += 0.5

    with placeholder.container():
        col1, col2 = st.columns([0.4, 0.6])

        col1.image(frame)
        col2.markdown("## **Status**: " + status)
        col2.markdown(f"## **Cont:** {int(cont)}")