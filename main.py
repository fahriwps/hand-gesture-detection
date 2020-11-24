import cv2
import imutils
import numpy as np
from sklearn.metrics import pairwise

bg = None

# Averaging frame and substraction
def run_avg(image, aWeight):
    global bg
    # Extracting frame background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # Calculating background frame againsts new frame with new objects
    cv2.accumulateWeighted(image, bg, aWeight)

# Segmentation and thresholding hand object
def segment(image, threshold=25):
    global bg
    # comparing difference background frame againsts new frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # thresholding new objects
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # calculating contour from thresholding 
    (_, cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 0:
        return
    else:
        # getting contour
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)


# Calculating how many fingers on contour
def count(thresholded, segmented):
	chull = cv2.convexHull(segmented)
	
	extreme_top    = tuple(chull[chull[:, :, 1].argmin()][0])
	extreme_bottom = tuple(chull[chull[:, :, 1].argmax()][0])
	extreme_left   = tuple(chull[chull[:, :, 0].argmin()][0])
	extreme_right  = tuple(chull[chull[:, :, 0].argmax()][0])

	cX = (extreme_left[0] + extreme_right[0]) // 2
	cY = (extreme_top[1] + extreme_bottom[1]) // 2

	distance = pairwise.euclidean_distances([(cX, cY)], Y=[extreme_left, extreme_right, extreme_top, extreme_bottom])[0]
	maximum_distance = distance[distance.argmax()]
	
	# calculating circle radius 
	radius = int(0.8 * maximum_distance)
	circumference = (2 * np.pi * radius)
	circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")

	# drawing cirle from new radius
	cv2.circle(circular_roi, (cX, cY), radius, 255, 1)
	
	# bitwise AND on circle
	circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)
	(_, cnts, _) = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	count = 0 

	for c in cnts:
		(x, y, w, h) = cv2.boundingRect(c)
		if ((cY + (cY * 0.25)) > (y + h)) and ((circumference * 0.25) > c.shape[0]):
			count += 1
	return count

if __name__ == "__main__":
    accumWeight = 0.5
    camera = cv2.VideoCapture(0)
    top, right, bottom, left = 10, 350, 225, 590
    num_frames = 0
    calibrated = False

    while(True):
        (grabbed, frame) = camera.read()
        frame = imutils.resize(frame, width=700)
        frame = cv2.flip(frame, 1)
        clone = frame.copy()
        (height, width) = frame.shape[:2]
        roi = frame[top:bottom, right:left]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        if num_frames < 30:
            run_avg(gray, accumWeight)
            if num_frames == 1:
            	print("[STATUS] sedang kalibrasi")
            elif num_frames == 29:
                print("[STATUS] kalibrasi sukses !")      
        else:
            hand = segment(gray)

            if hand is not None:
                (thresholded, segmented) = hand
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                fingers = count(thresholded, segmented)
                cv2.putText(clone, str(fingers), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.imshow("Thesholded", thresholded)

        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

        num_frames += 1

        cv2.imshow("Video Stream", clone)

        keypress = cv2.waitKey(1) & 0xFF

        if keypress == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()
