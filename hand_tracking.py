import cv2
import mediapipe as mp
import time
import logging

class HandDetector:
    def __init__(self, mode=False, max_hands=2, model_complexity=1, detect_confidence=0.5, track_confidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.model_complexity = model_complexity
        self.detect_confidence = detect_confidence
        self.track_confidence = track_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands, self.model_complexity, self.detect_confidence, self.track_confidence)
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):

        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(image_rgb)
        # print(results.multi_hand_landmarks)

        # show hand landmarks in display
        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:

                if draw:
                    self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)

        return img

    def find_position(self, img, hand_number=0, draw=True):

        lm_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_number]

            for id, lm in enumerate(my_hand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)

                lm_list.append([id, cx, cy])

                # draw filled pink circles on all hand landmarks
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        return lm_list

def main():
    prev_time = 0
    cur_time = 0

    detector = HandDetector()

    # set camera to default
    cap = cv2.VideoCapture(0)

    # check if camera opened
    if not cap.isOpened():
        logging.error("No camera opened")
        exit(1)

    while True:
        success, img = cap.read()

        img = detector.find_hands(img)

        lm_list = detector.find_position(img)

        if len(lm_list) != 0:
            print(lm_list[4])

        # flip camera
        img = cv2.flip(img, 1)

        cur_time = time.time()
        fps = 1 / (cur_time - prev_time)
        prev_time = cur_time

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_ITALIC, 2, (255, 0, 255), 3)

        # display image
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()