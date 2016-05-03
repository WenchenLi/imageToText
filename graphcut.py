# graphcut.py
import numpy as np
import scipy
from scipy.misc import imread
import maxflow
import cv2

def graphcut_binary(cv_img):
    # img = imread("data/scenetext06.jpg")
    img = cv_img[::-1,:,::-1]
    # Create the graph.
    g = maxflow.Graph[int]()
    # Add the nodes. nodeids has the identifiers of the nodes in the grid.
    nodeids = g.add_grid_nodes(img.shape)
    # Add non-terminal edges with the same capacity.
    g.add_grid_edges(nodeids, 50)
    # Add the terminal edges. The image pixels are the capacities
    # of the edges from the source node. The inverted image pixels
    # are the capacities of the edges to the sink node.
    g.add_grid_tedges(nodeids, img, 255 - img)

    # Find the maximum flow.
    g.maxflow()
    # Get the segments of the nodes in the grid.
    sgm = g.get_grid_segments(nodeids)

    # The labels should be 1 where sgm is False and 0 otherwise.
    img2 = np.int_(np.logical_not(sgm))
    # Show the result.
    from matplotlib import pyplot as ppl
    ppl.imshow(img2)
    ppl.show()
    return img2[::-1,:,::-1]

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    gen = (i for i in range(100))
    while (True):
        ret, frame = cap.read()
        cv2.imwrite("frame"+str(gen.next())+".jpg",frame)
        # cv2.imshow('frame_graphcut', graphcut_binary(frame))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
