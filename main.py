import cv2
import imutils
import numpy as np
from sklearn.metrics import pairwise

bg = None


def averaging_frame(image, weight):
    """Averaging frame by calculating background
    frame againsts new frame.

    Args:
      image: An image to be calculated.
      weight: Accumulate weight for thresholding.

    Return:
      bg: Image copy as background if `bg` is None.
    """
    global bg
    if bg is None:
        bg = image.copy().astype("float")
        return bg

    cv2.accumulateWeighted(image, bg, weight)


def segmenting_object(image, threshold=25):
    """Segmentation and threshold hand object from
    new frame against background frame.

    Args:
      image: Image to be calculated.
      threshold: Value to segment object from the background.

    Returns:
      thresholded_image: Image after threshold operation.
      segmented: Image with segmented object.
    """
    global bg
    diff = cv2.absdiff(bg.astype("uint8"), image)
    thresholded_image = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    _, cnts, _ = cv2.findContours(
        thresholded_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(cnts) == 0:
        return

    else:
        segmented = max(cnts, key=cv2.contourArea)
        return thresholded_image, segmented


def counting_finger(thresholded, segmented):
    """Main function for counting fingers from images.

    Args:
      thresholded: Image after threshold operation.
      segmented: Image with segmented object.

    Return:
      counter: Finger counter value.
    """
    chull = cv2.convexHull(segmented)

    extreme_top = tuple(chull[chull[:, :, 1].argmin()][0])
    extreme_bottom = tuple(chull[chull[:, :, 1].argmax()][0])
    extreme_left = tuple(chull[chull[:, :, 0].argmin()][0])
    extreme_right = tuple(chull[chull[:, :, 0].argmax()][0])

    cX = (extreme_left[0] + extreme_right[0]) // 2
    cY = (extreme_top[1] + extreme_bottom[1]) // 2

    distance = pairwise.euclidean_distances(
        [(cX, cY)], Y=[extreme_left, extreme_right, extreme_top, extreme_bottom]
    )[0]
    maximum_distance = distance[distance.argmax()]

    circle_radius = int(0.8 * maximum_distance)
    circumference = 2 * np.pi * circle_radius
    circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")

    cv2.circle(circular_roi, (cX, cY), circle_radius, 255, 1)

    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)
    _, cnts, _ = cv2.findContours(
        circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    counter = 0

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if ((cY + (cY * 0.25)) > (y + h)) and ((circumference * 0.25) > c.shape[0]):
            counter += 1
    return counter


if __name__ == "__main__":
    accumWeight = 0.5
    camera = cv2.VideoCapture(0)
    if camera is None or not camera.isOpened():
        raise ValueError("UNABLE TO OPEN CAM")

    top, right, bottom, left = 10, 350, 225, 590
    num_frames = 0
    calibrated = False

    while True:
        grabbed, frame = camera.read()
        frame = imutils.resize(frame, width=700)
        frame = cv2.flip(frame, 1)
        clone = frame.copy()
        height, width = frame.shape[:2]
        roi = frame[top:bottom, right:left]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        if num_frames < 30:
            averaging_frame(gray, accumWeight)
            if num_frames == 1:
                print("[STATUS] sedang kalibrasi")
            elif num_frames == 29:
                print("[STATUS] kalibrasi sukses !")
        else:
            hand = segmenting_object(gray)

            if hand is not None:
                (thresholded, segmented) = hand
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                fingers = counting_finger(thresholded, segmented)
                cv2.putText(
                    clone,
                    str(fingers),
                    (70, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
                cv2.imshow("Thesholded", thresholded)

        cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)

        num_frames += 1

        cv2.imshow("Video Stream", clone)

        keypress = cv2.waitKey(1) & 0xFF

        if keypress == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
