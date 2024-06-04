from djitellopy import Tello
import cv2
import numpy as np
import mediapipe as mp
import time

# Initializing MediaPipe Hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Frame source: 0 -> from webcam, 1 -> from drone



teclado = True
automatico = False

in_speed = 20
in_height = 50

# Initializing camera stream

capture = cv2.VideoCapture(0)

drone = Tello()
drone.connect()
drone.streamoff()
drone.streamon()
drone.speed = 20
drone.left_right_velocity = 0
drone.forward_backward_velocity = 0
drone.up_down_velocity = 0
drone.yaw_velocity = 0
drone.status = 0

# Image size
h = 500
w = 500

def update_speed(value):
    global drone
    drone.speed = value

def TrackAltura(value):
    global drone
    target_height = value
    current_height = drone.get_height()
    
    if target_height > current_height:
        move_value = target_height - current_height
        if move_value < 20:
            move_value = 20
        drone.move_up(move_value)
    elif target_height < current_height:
        move_value = current_height - target_height
        if move_value < 20:
            move_value = 20
        drone.move_down(move_value)

cv2.namedWindow('Trackbars')
cv2.resizeWindow('Trackbars', (500, 100))

cv2.createTrackbar('Speed', 'Trackbars', 0, 100, update_speed)
cv2.createTrackbar('Height', 'Trackbars', 50, 300, TrackAltura)
cv2.setTrackbarPos('Speed', 'Trackbars', in_speed)
cv2.setTrackbarPos('Height', 'Trackbars', in_height)

def is_handR_open(landmarks):
    fingers_extended = all([
        landmarks[mp_hands.HandLandmark.THUMB_TIP].y < landmarks[mp_hands.HandLandmark.THUMB_IP].y,
        landmarks[mp_hands.HandLandmark.THUMB_TIP].x < landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].x,
        landmarks[mp_hands.HandLandmark.PINKY_TIP].y > landmarks[mp_hands.HandLandmark.RING_FINGER_DIP].y,
        landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].y,
        landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y,
        landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.RING_FINGER_MCP].y,
        landmarks[mp_hands.HandLandmark.PINKY_TIP].y < landmarks[mp_hands.HandLandmark.PINKY_MCP].y
    ])
    
    return fingers_extended

def is_handL_open(landmarks):
    fingers_extended = all([
        landmarks[mp_hands.HandLandmark.THUMB_TIP].y < landmarks[mp_hands.HandLandmark.THUMB_IP].y,
        landmarks[mp_hands.HandLandmark.THUMB_TIP].x > landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].x,
        landmarks[mp_hands.HandLandmark.PINKY_TIP].x < landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].x,
        landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].y,
        landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y,
        landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.RING_FINGER_MCP].y,
        landmarks[mp_hands.HandLandmark.PINKY_TIP].y < landmarks[mp_hands.HandLandmark.PINKY_MCP].y
    ])
    
    return fingers_extended

def hand_right(landmarks):
    fingers = all([
        landmarks[mp_hands.HandLandmark.THUMB_TIP].x > landmarks[mp_hands.HandLandmark.THUMB_IP].x,
        landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x > landmarks[mp_hands.HandLandmark.INDEX_FINGER_DIP].x,
        landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x > landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x,
        landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].x > landmarks[mp_hands.HandLandmark.RING_FINGER_DIP].x,
        landmarks[mp_hands.HandLandmark.PINKY_TIP].x > landmarks[mp_hands.HandLandmark.PINKY_DIP].x,
        landmarks[mp_hands.HandLandmark.PINKY_TIP].y < landmarks[mp_hands.HandLandmark.PINKY_DIP].y,
    ])

    return fingers

def hand_left(landmarks):
    fingers = all([
        landmarks[mp_hands.HandLandmark.THUMB_TIP].x < landmarks[mp_hands.HandLandmark.THUMB_IP].x,
        landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x < landmarks[mp_hands.HandLandmark.INDEX_FINGER_DIP].x,
        landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x < landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x,
        landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].x < landmarks[mp_hands.HandLandmark.RING_FINGER_DIP].x,
        landmarks[mp_hands.HandLandmark.PINKY_TIP].x < landmarks[mp_hands.HandLandmark.PINKY_DIP].x,
        landmarks[mp_hands.HandLandmark.PINKY_TIP].y < landmarks[mp_hands.HandLandmark.PINKY_DIP].y,
    ])

    return fingers

def index_middle_up(landmarks):
    index_up = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.INDEX_FINGER_DIP].y 
    middle_up = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y 
    fingers_closed = all([
        landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y > landmarks[mp_hands.HandLandmark.RING_FINGER_MCP].y,
        landmarks[mp_hands.HandLandmark.PINKY_TIP].y > landmarks[mp_hands.HandLandmark.PINKY_MCP].y
    ])
    
    return index_up and fingers_closed and middle_up

def index_up(landmarks):
    index_up = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.INDEX_FINGER_DIP].y 
    fingers_closed = all([
        landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y,
        landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y > landmarks[mp_hands.HandLandmark.RING_FINGER_MCP].y,
        landmarks[mp_hands.HandLandmark.PINKY_TIP].y > landmarks[mp_hands.HandLandmark.PINKY_MCP].y
    ])
    
    return index_up and fingers_closed

def closed_fist(landmarks):
    fingers_closed = all([
        landmarks[mp_hands.HandLandmark.THUMB_TIP].y > landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y,
        landmarks[mp_hands.HandLandmark.THUMB_TIP].y < landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y,
        landmarks[mp_hands.HandLandmark.THUMB_TIP].x > landmarks[mp_hands.HandLandmark.INDEX_FINGER_DIP].x,
        landmarks[mp_hands.HandLandmark.THUMB_TIP].x < landmarks[mp_hands.HandLandmark.RING_FINGER_PIP].x,
        landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y > landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].y,
        landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y,
        landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y > landmarks[mp_hands.HandLandmark.RING_FINGER_MCP].y,
        landmarks[mp_hands.HandLandmark.PINKY_TIP].y > landmarks[mp_hands.HandLandmark.PINKY_MCP].y
    ])
    
    return fingers_closed

def is_thumb_up_with_fist(landmarks):
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP].y
    thumb_up = thumb_tip < landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
    fingers_closed = all([
        landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y > landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].y,
        landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y,
        landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y > landmarks[mp_hands.HandLandmark.RING_FINGER_MCP].y,
        landmarks[mp_hands.HandLandmark.PINKY_TIP].y > landmarks[mp_hands.HandLandmark.PINKY_MCP].y
    ])
    
    return thumb_up and fingers_closed

def is_thumb_down_with_fist(landmarks):
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP].y
    thumb_ip = landmarks[mp_hands.HandLandmark.THUMB_IP].y
    thumb_down = thumb_tip > thumb_ip
    fingers_closed = all([
        landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y > landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].y,
        landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y,
        landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y > landmarks[mp_hands.HandLandmark.RING_FINGER_MCP].y,
        landmarks[mp_hands.HandLandmark.PINKY_TIP].y > landmarks[mp_hands.HandLandmark.PINKY_MCP].y
    ])
    return thumb_down and fingers_closed

def thumb_pinky_up(landmarks):
    thumb_up = landmarks[mp_hands.HandLandmark.THUMB_TIP].y < landmarks[mp_hands.HandLandmark.THUMB_IP].y
    pinky_up = landmarks[mp_hands.HandLandmark.PINKY_TIP].y < landmarks[mp_hands.HandLandmark.PINKY_DIP].y
    other_fingers_closed = all([
        landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y > landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].y,
        landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y,
        landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y > landmarks[mp_hands.HandLandmark.RING_FINGER_MCP].y
    ])
    return thumb_up and pinky_up and other_fingers_closed

def four_fingers_up(landmarks):
    index_up = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.INDEX_FINGER_DIP].y 
    middle_up = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y 
    ring_up = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.RING_FINGER_DIP].y
    pinky_up = landmarks[mp_hands.HandLandmark.PINKY_TIP].y < landmarks[mp_hands.HandLandmark.PINKY_DIP].y
    other_fingers_closed = all([
        landmarks[mp_hands.HandLandmark.THUMB_TIP].x > landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x,
        landmarks[mp_hands.HandLandmark.THUMB_TIP].x < landmarks[mp_hands.HandLandmark.PINKY_MCP].x,
        landmarks[mp_hands.HandLandmark.PINKY_TIP].y > landmarks[mp_hands.HandLandmark.RING_FINGER_DIP].y
    ])
    return index_up and middle_up and ring_up and other_fingers_closed

def main():
    global automatico, teclado
    with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        print("main program running now")
        last_detection_time = time.time()
        detection_timeout = 2  # segundos
        
        while True:
            
            ret, img2 = capture.read()
            
            frame_read = drone.get_frame_read()
            img = frame_read.frame
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            img = cv2.flip(img, 1)
            img = cv2.resize(img, (500, 500))
            

            #img2 = frame_read.frame
            
            img2 = cv2.flip(img2, 1)
            
            img2 = cv2.resize(img2, (500, 500))
            #img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
            results = hands.process(img2)






            key = cv2.waitKey(15) & 0xFF
            
            if key == 112:
                if teclado:
                    automatico = True
                    teclado = False
                else:
                    automatico = False
                    teclado = True
                    
            if teclado and not automatico:
                cv2.putText(img2, 'Teclado', 
                            (400, 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)
            if not teclado and automatico:
                cv2.putText(img2, 'Automatico', 
                            (400, 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1) 

            if results.multi_hand_landmarks:
                last_detection_time = time.time()
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(img2, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    if is_thumb_up_with_fist(hand_landmarks.landmark):
                        cv2.putText(img2, "Pulgar arriba con punio cerrado", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                                    cv2.LINE_AA)
                        if automatico: drone.takeoff()
                   
                    if is_thumb_down_with_fist(hand_landmarks.landmark):
                        cv2.putText(img2, "Pulgar abajo con punio cerrado", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                                    cv2.LINE_AA)
                        if automatico: drone.land()
                    if is_handR_open(hand_landmarks.landmark):
                        cv2.putText(img2, "Mano derecha abierta", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                                    cv2.LINE_AA)
                        if  automatico: drone.send_rc_control(0, 0, 0, 0)
                    if closed_fist(hand_landmarks.landmark):
                        cv2.putText(img2, "punio cerrado", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                                    cv2.LINE_AA)
                        if automatico: drone.send_rc_control(0, 0, 0, drone.speed)
                    if index_up(hand_landmarks.landmark):
                        cv2.putText(img2, "index levantado", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                                    cv2.LINE_AA)
                        if  automatico: drone.send_rc_control(0, drone.speed, 0, 0)
                    if index_middle_up(hand_landmarks.landmark):
                        cv2.putText(img2, "index y middle levantado", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                                    cv2.LINE_AA)
                        if  automatico: drone.send_rc_control(0, -drone.speed, 0, 0)
                    if hand_right(hand_landmarks.landmark):
                        cv2.putText(img2, "mano hacia derecha", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                                    cv2.LINE_AA)
                        if automatico: drone.send_rc_control(-drone.speed, 0, 0, 0)
                    if hand_left(hand_landmarks.landmark):
                        cv2.putText(img2, "mano hacia izquiera", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                                    cv2.LINE_AA)
                        if  automatico: drone.send_rc_control(drone.speed, 0, 0, 0)
                    if is_handL_open(hand_landmarks.landmark):
                        cv2.putText(img2, "Mano izquierda abierta", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                                    cv2.LINE_AA)
                        if automatico: drone.send_rc_control(0, 0, -drone.speed, 0)
                    if thumb_pinky_up(hand_landmarks.landmark):
                        cv2.putText(img2, "Pulgar y menique levantados", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                                    cv2.LINE_AA)
                        if automatico: drone.send_rc_control(0, 0, 0, -drone.speed)
                    if four_fingers_up(hand_landmarks.landmark):
                        cv2.putText(img2, "cuatro dedos levantados", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                                    cv2.LINE_AA)
                        if  automatico: drone.send_rc_control(0, 0, drone.speed, 0)
            else:
                if time.time() - last_detection_time > detection_timeout:
                    if automatico:
                        drone.send_rc_control(0, 0, 0, 0)
                        cv2.putText(img2, 'No hand detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                        
            
            cv2.putText(img, 'Battery: ' + str(drone.get_battery()), (5, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow('image', img)
            cv2.imshow('image2', img2)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if teclado:
                if key == 116:
                    if drone.get_battery() >= 5:
                        if drone.status == 0:
                            drone.status = 1
                            drone.takeoff()
                    else:
                        img2 = cv2.imread('fondo.jpg')
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(img2, "Bateria baja para despegue", (10, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.imshow("Warning", img2)        
                                
                if key == 108: 
                    if drone.status == 1: 
                        drone.status = 0
                        drone.land()

                elif key == 104: 
                    drone.left_right_velocity = 0 
                    drone.forward_backward_velocity = 0
                    drone.up_down_velocity = 0
                    drone.yaw_velocity = 0

                elif key == 119: 
                    drone.left_right_velocity = 0 
                    drone.forward_backward_velocity = drone.speed
                    drone.up_down_velocity = 0
                    drone.yaw_velocity = 0 

                elif key == 115: 
                    drone.left_right_velocity = 0 
                    drone.forward_backward_velocity = -drone.speed
                    drone.up_down_velocity = 0
                    drone.yaw_velocity = 0
                
                elif key == 97: 
                    drone.left_right_velocity = -drone.speed
                    drone.forward_backward_velocity = 0
                    drone.up_down_velocity = 0
                    drone.yaw_velocity = 0 

                elif key == 100: 
                    drone.left_right_velocity = drone.speed
                    drone.forward_backward_velocity = 0
                    drone.up_down_velocity = 0
                    drone.yaw_velocity = 0

                elif key == 122: 
                    drone.left_right_velocity = 0
                    drone.forward_backward_velocity = 0
                    drone.up_down_velocity = 0
                    drone.yaw_velocity = -drone.speed

                elif key == 120: 
                    drone.left_right_velocity = 0
                    drone.forward_backward_velocity = 0
                    drone.up_down_velocity = 0
                    drone.yaw_velocity = drone.speed

                elif key == 101: 
                    drone.left_right_velocity = 0
                    drone.forward_backward_velocity = 0
                    drone.up_down_velocity = drone.speed
                    drone.yaw_velocity = 0

                elif key == 114: 
                    drone.left_right_velocity = 0
                    drone.forward_backward_velocity = 0
                    drone.up_down_velocity = -drone.speed
                    drone.yaw_velocity = 0

                drone.send_rc_control(drone.left_right_velocity,
                                drone.forward_backward_velocity,
                                drone.up_down_velocity,
                                drone.yaw_velocity)

    
    capture.release()
    
    drone.streamoff()
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
