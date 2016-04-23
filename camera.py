import numpy as np
import cv2

try:
    import Image
except ImportError:
    from PIL import Image
import pytesseract
import enchant

cap = cv2.VideoCapture(0)
d = enchant.Dict("en_US")

while (True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = Image.fromarray(gray)
    res = pytesseract.image_to_string(img, boxes=False, lang='eng')
    res = res.split()
    if res:
        for word in res: d.suggest(word)
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):break
cap.release()
cv2.destroyAllWindows()
