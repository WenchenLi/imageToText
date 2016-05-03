import cv2

try:
    import Image
except ImportError:
    from PIL import Image
import pytesseract
import enchant
from grabcut import *

font = cv2.FONT_HERSHEY_SIMPLEX
cap = cv2.VideoCapture(0)
d = enchant.Dict("en_US")
box = True
while (True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    color = cv2.cvtColor(frame, 0)
    frame = gray  # cv2.medianBlur(gray, 3)
    th2 = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 11, 3)
    # th3 = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                             cv2.THRESH_BINARY, 11, 2)
    img = Image.fromarray(th2)
    res = pytesseract.image_to_string(img, boxes=box, lang='eng')
    res_list = res.split('\n')
    # for i in res_list:
    #     if i and not i[0].isalnum():
    #         res_list.remove(i)
    if box and res_list:
        hi, wi = gray.shape[0], gray.shape[1]
        for c in res_list:
            c = c.split()
            if c:
                x1, y1, x2, y2 = int(
                    c[1]), hi - int(c[2]), int(c[3]), int(c[4])
                cv2.putText(color, c[0], (x1, y1), 0, 1, (0, 255, 0), 2)
        cv2.imshow("Show", color)
    res = pytesseract.image_to_string(img,lang='eng')
    suggested = [d.suggest(w)[0] for w in res.split() if d.suggest(w)]
    print " ".join(suggested)
    # cv2.putText(color, " ".join(suggested), (20, 20), 0, 1, (0, 0, 255), 2)
    # cv2.imshow('color', color)

    cv2.imshow('frame', th2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
