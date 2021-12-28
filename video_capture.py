from keras.models import load_model
import cv2
import numpy as np

model = load_model("VGG19-Face Mask Detection.h5")
face_clsfr = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
print("here")
source = cv2.VideoCapture(1)
cv2.startWindowThread()
labels_dict = {0: 'with_mask', 1: 'without_mask'}
color_dict = {0: (0, 255, 0), 1: (0, 0, 255)}


def get_result(resized):
    img = np.array(resized).reshape([1, 128, 128, 3]) / 255.0

    if ((model.predict(img) > 0.5).astype("int32")) == 0:
        return 0
    else:
        return 1


while (True):

    ret, img = source.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_clsfr.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        face_img = img[y:y + w, x:x + w]
        face_img = cv2.resize(face_img, (128, 128))
        face_img = np.reshape(face_img, [1, 128, 128, 3]) / 255.0
        mask_result = model.predict(face_img).argmax()
        cv2.rectangle(img, (x, y), (x+w, y+h), color_dict[mask_result],1 )
        # cv2.rectangle(img, (x, y), (x + w, y + h), color_dict[mask_result], 2)
        # cv2.rectangle(img, (x, y - 40), (x + w, y), color_dict[mask_result], -1)
        cv2.putText(
            img, labels_dict[mask_result],
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('LIVE', img)
    key = cv2.waitKey(1)

    if (key == 27):
        break

cv2.destroyAllWindows()
source.release()
