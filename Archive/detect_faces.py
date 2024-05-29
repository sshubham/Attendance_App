# face_detector.py

import cv2
import time

class FaceDetector:
    def __init__(self, cascade_path='model/haarcascade_frontalface_default.xml'):
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.cap = None

    def start_detection(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640)  # set Width
        self.cap.set(4, 480)  # set Height

        while True:
            ret, img = self.cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=10,
                minSize=(20, 20)
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

            cv2.imshow('video', img)

            k = cv2.waitKey(30) & 0xff
            if k == 27:  # press 'ESC' to quit
                break

    def stop_detection(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = FaceDetector()
    try:
        detector.start_detection()
        time.sleep(1000)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        detector.stop_detection()