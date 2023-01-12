import cv2 as cv
import mediapipe as mp
from cvzone import HandTrackingModule

cap = cv.VideoCapture(0)


# using cvzone (built with opencv and mediapipe)
def cvzone_hand_detection_frame():
    detector = HandTrackingModule.HandDetector()

    while True:
        ret, img = cap.read()
        hands, img = detector.findHands(img)
        if hands:
            cv.putText(img, f'number of hands: {len(hands)}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1,
                       (0, 0, 0), 2, cv.LINE_AA)

        cv.imshow("Hand Detection frame", img)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


def mediapipe_hand_raise_detection():
    mp_drawing = mp.solutions.drawing_utils
    # mp_drawing_styles = mp.solutions.drawing_styles

    # model initialization
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    ret, img = cap.read()
    h, w, c = img.shape

    while True:
        ret, img = cap.read()
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        results = hands.process(img)

        img.flags.writeable = True
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                x_max = 0
                y_max = 0
                x_min = w
                y_min = h
                for lm in hand_lms.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    if x > x_max:
                        x_max = x
                    if x < x_min:
                        x_min = x
                    if y > y_max:
                        y_max = y
                    if y < y_min:
                        y_min = y
                box = cv.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv.putText(box, 'hand detected', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA)
                mp_drawing.draw_landmarks(
                    img,
                    hand_lms
                    # mp_hands.HAND_CONNECTIONS,
                    # mp_drawing_styles.get_default_hand_landmarks_style(),
                    # mp_drawing_styles.get_default_hand_connections_style()
                )
        cv.imshow("Mediapipe hands", img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()


# cvzone_hand_detection_frame()
# mediapipe_hand_raise_detection()
