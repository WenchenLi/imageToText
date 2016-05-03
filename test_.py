try:
    import Image
except ImportError:
    from PIL import Image
import pytesseract
import cv2
import random

filename = 'data/test_my.jpg'
npimg = cv2.imread(filename, 1)
box = True
img = Image.fromarray(npimg)
res = pytesseract.image_to_string(img, boxes=box)
res_list = res.split('\n')
print pytesseract.image_to_string(img)

if box and res_list:
    # res_list = res.split('\n')
    hi, wi = npimg.shape[0], npimg.shape[1]
    for c in res_list:
        c = c.split()
        # print c
        x1, y1, x2, y2 = int(c[1]), hi - int(c[2]), int(c[3]), int(c[4])
        # cv2.rectangle(npimg, (x1, y1), (x2, y2), (0, 255, 255), 1)
        cv2.putText(npimg, c[0], (x1, y1), 0, 1, (0, 255, 0), 2)
    cv2.imshow("Show", npimg)
    cv2.imwrite('res.png', npimg)
    cv2.waitKey()
    cv2.destroyAllWindows()
