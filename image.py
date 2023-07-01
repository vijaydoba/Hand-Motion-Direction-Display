import cv2
import mediapipe as mp

# Initialize Mediapipe Hand class
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Open the camera
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    if not ret:
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe
    results = mp_hands.process(rgb_frame)

    # Check if hand is detected
    if results.multi_hand_landmarks:
        # Get the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]

        # Get the X-coordinate of the index finger tip
        index_finger_x = hand_landmarks.landmark[8].x

        # Get the X-coordinate of the thumb tip
        thumb_x = hand_landmarks.landmark[4].x

        # Determine the direction based on finger positions
        if index_finger_x < thumb_x:
            direction = "Left"
        else:
            direction = "Right"

        # Display the direction
        cv2.putText(frame, direction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        # Display "Please move your hand"
        cv2.putText(frame, "Please move your hand", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Hand Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
