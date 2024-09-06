import cv2
import mediapipe as mp
import math

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

def calculate_angle(a, b, c):
    ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    return abs(ang) if abs(ang) < 180 else 360 - abs(ang)

cap = cv2.VideoCapture('test.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Pose detection
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        # Draw the landmarks on the image
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Get the height and width of the image
        h, w, _ = frame.shape
        
        # Extract the coordinates of key points (hips, knees, ankles, shoulders, elbows)
        landmarks = results.pose_landmarks.landmark
        
        # Coordinates for analyzing stride and posture
        right_hip = (int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * w), 
                    int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * h))
        right_knee = (int(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x * w), 
                    int(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y * h))
        right_ankle = (int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x * w), 
                    int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y * h))

        left_hip = (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * w), 
                    int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * h))
        left_knee = (int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x * w), 
                    int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y * h))
        left_ankle = (int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x * w), 
                    int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y * h))
        
        # Coordinates of shoulders and torso
        right_shoulder = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w), 
                        int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h))
        left_shoulder = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w), 
                        int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h))
        nose = (int(landmarks[mp_pose.PoseLandmark.NOSE].x * w), 
                int(landmarks[mp_pose.PoseLandmark.NOSE].y * h))

        # 1 - Stride length (distance between ankles)
        stride_length = calculate_distance(right_ankle, left_ankle)
        print(f'Stride length: {stride_length}px')

        # 2 - Right knee angle (hip, knee, ankle)
        knee_angle_right = calculate_angle(right_hip, right_knee, right_ankle)
        print(f'Right knee angle: {knee_angle_right:.2f}°')

        # 3 - Left knee angle (hip, knee, ankle)
        knee_angle_left = calculate_angle(left_hip, left_knee, left_ankle)
        print(f'Left knee angle: {knee_angle_left:.2f}°')

        # 4 - Torso angle (right shoulder, left shoulder, hip)
        torso_angle = calculate_angle(right_shoulder, left_shoulder, left_hip)
        print(f'Torso angle: {torso_angle:.2f}°')

        # 5 - Right foot strike (angle between right ankle and the ground)
        ground_level = (right_ankle[0], h)
        ankle_attack_angle_right = calculate_angle(ground_level, right_ankle, right_knee)
        print(f'Right ankle attack angle: {ankle_attack_angle_right:.2f}°')

        # 6 - Left foot strike (angle between left ankle and the ground)
        ground_level_left = (left_ankle[0], h)
        ankle_attack_angle_left = calculate_angle(ground_level_left, left_ankle, left_knee)
        print(f'Left ankle attack angle: {ankle_attack_angle_left:.2f}°')

        # 7 - Shoulder symmetry (height difference between right and left shoulders)
        shoulder_symmetry = abs(right_shoulder[1] - left_shoulder[1])
        print(f'Shoulder height difference: {shoulder_symmetry}px')

    # Display the video with annotations
    cv2.imshow('Stride Analysis', frame)

    if cv2.waitKey(200) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
