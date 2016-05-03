import numpy as np
import cv2
from matplotlib import pyplot as plt


class GrabCut(object):
    def __init__(self):
        self.mask = None#np.zeros(img.shape[:2], np.uint8)
        self.bgdModel = bgdModel = np.zeros((1, 65), np.float64)
        self.fgdModel = np.zeros((1, 65), np.float64)
        self.rect = (50, 50, 600, 480)
        self.iter = 2
        self.count = 0
    
    def grabcut(self, frame):
        img = frame
        self.mask = np.zeros(img.shape[:2], np.uint8)

        if self.count %2 == 0: 
            cv2.grabCut(img, self.mask, self.rect, self.bgdModel, self.fgdModel, self.iter,cv2.GC_INIT_WITH_RECT)
        else:cv2.grabCut(img, self.mask, None, self.bgdModel, self.fgdModel, self.iter,cv2.GC_INIT_WITH_MASK)
        # self.count += 1

        mask2 = np.where((self.mask == 2) | (self.mask == 0), 0, 1).astype('uint8')
        img = img * mask2[:, :, np.newaxis]

        # plt.imshow(mask2),plt.colorbar(),plt.show()
        return img


if __name__ == '__main__':
    gc = GrabCut()
    cap = cv2.VideoCapture(-1)
    while (cap.isOpened()):
        ret, frame = cap.read()
        cv2.imshow('img', gc.grabcut(frame))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
