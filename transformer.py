import numpy as np
import cv2
'credit for order_points and four_point_transform:pyimagesearch'

def get_corner_manual(filename):
	print 'please go from top left to top right to bottom right then bottom left'
	pts = []
	img = cv2.imread(filename)

	# mouse callback function
	def draw_circle(event, x, y, flags, param):
		if event == cv2.EVENT_LBUTTONDOWN:
			cv2.circle(img, (x, y), 1, (0, 255, 0), -1)
			pts.append((x, y))

	# Create a black image, a window and bind the function to window
	cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
	cv2.setMouseCallback('image', draw_circle)

	while (len(pts) < 4):
		cv2.imshow('image', img)
		if cv2.waitKey(20) & 0xFF == 27:
			break
	cv2.destroyAllWindows()
	print pts
	return pts


def get_corners(filename, threshold):
	img = cv2.imread(filename)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	gray = np.float32(gray)
	dst = cv2.cornerHarris(gray, 2, 3, 0.04)
	# result is dilated for marking the corners, not important
	dst = cv2.dilate(dst, None)

	# Threshold for an optimal value, it may vary depending on the image.
	img[dst > threshold * dst.max()] = [0, 0, 255]

	cv2.imwrite(filename.replace('.jpg', '_corner.jpg'), img)
	if cv2.waitKey(0) & 0xff == 27:
		cv2.destroyAllWindows()


def get_corners_subpixel(filename):
	img = cv2.imread(filename)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# find Harris corners
	gray = np.float32(gray)
	dst = cv2.cornerHarris(gray, 2, 3, 0.04)
	dst = cv2.dilate(dst, None)
	ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
	dst = np.uint8(dst)

	# find centroids
	ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

	# define the criteria to stop and refine the corners
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
	corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

	# Now draw them
	res = np.hstack((centroids, corners))
	res = np.int0(res)
	img[res[:, 1], res[:, 0]] = [0, 0, 255]
	img[res[:, 3], res[:, 2]] = [0, 255, 0]

	cv2.imwrite(filename.replace('.jpg', '_corner_sub.jpg'), img)


def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype="float32")

	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis=1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis=1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	# return the ordered coordinates
	return rect


def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype="float32")

	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	return warped


if __name__ == '__main__':
	# get_corner_manual('data/street.jpg')

	pts =[(110, 118), (883, 220), (892, 552), (87, 552)]

	image = cv2.imread('data/street.jpg',1)
	pts = np.array(pts, dtype = "float32")
	warped = four_point_transform(image, pts)
	cv2.imshow("Original", image)
	cv2.imshow("Warped", warped)
	cv2.imwrite('data/street_warped.jpg', warped)
	cv2.waitKey(0)
