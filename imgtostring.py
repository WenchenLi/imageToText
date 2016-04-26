try:
    import Image
except ImportError:
    from PIL import Image
import pytesseract
import cv2
import random

from detector import *
from transformer import *
from collections import *
#TODO1: scale,2: sliding window
#imagine image is a platue initially, search based on score and search give score feedback, for the area,
#have char + 1, non -1. 
filename = 'data_resize/scenetext_segmented_word01.jpg'

candidate_pts = {}
bow = defaultdict(list)
# detector = Detector(filename)
# detector.detect_text(visual=True)
# detector.visualize()
# rects = detector.getRects()['text']
# rects.append(detector.getTextBlocks())
# for r in rects:
#     candidate_pts[tuple(r)] = get_candidate_corners(
#         get_corners(filename, .01), tuple(r))
# corners = get_corners(filename, .01)
pts_gen = sliding_window(100,(128,256))#for sliding window, if true, stiching together

string = None
for i in xrange(100):
    pts = pts_gen.next()
    # r = random.choice(candidate_pts.keys())
    # for k in candidate_pts[r]:
    #     pts.append(random.choice(candidate_pts[r][k]))
    # print pts
    # for i in xrange(4): pts.append(random.choice(corners))
    image = cv2.imread(filename, 1)
    pts = np.array(pts, dtype="float32")
    pts_img = image.copy()
    for p in pts: cv2.circle(pts_img,tuple(p),2 , (0,255,0),5)
    cv2.imshow("pts", pts_img)
    cv2.waitKey(1)
    warped = four_point_transform(image, pts)
    # cv2.imshow("Warped", warped)
    # cv2.waitKey(0)
    box = True
    npimg = warped
    img = Image.fromarray(npimg)
    res = pytesseract.image_to_string(img, boxes=box)
    res_list = res.split('\n')
    for i in res_list:
        if i and not i[0].isalnum():
            res_list.remove(i)
    if res_list == [''] or []:
        continue
    string = pytesseract.image_to_string(img)
    s = string.split()
    print s
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
