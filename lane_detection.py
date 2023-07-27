import cv2
import numpy as np


def region_of_interest(image, region_points):

    # first replacing the "not required region" with 0(black) pixels
    mask = np.zeros_like(image)
    # then replacing the "required region" with 255(white) pixels
    cv2.fillPoly(mask, region_points, 255)
    # on a black image (because of mask) we are replacing the required region with 255(white) pixels.
    # keeping the part of image from the "required region"
    masked_image = cv2.bitwise_and(image, mask)  # here the image is Canny's kernel modified image
    # therefore the image leaving the required region would all become black.
    return masked_image


def draw_the_lines(image, lines):

    # creating a different image for the lines
    lines_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    # there are (x,y) as the starting and end points of the lines
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), thickness=6)  # BGR

    # merging the image with the lines
    image_with_lines = cv2.addWeighted(image, 0.6, lines_image, 1, 0.0)

    return image_with_lines


def get_lane_detected(image):

    (height, width) = (image.shape[0], image.shape[1])

    # converting the image into grayscale image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # edge detection kernel (Canny's algorithm)
    canny_image = cv2.Canny(gray_image, 100, 120)  # 100 is the lower threshold and 120 is the higher threshold

    # end of gray color converter function.

    region_of_interest_vertices = [
                (0, height),
                (width*0.5, height*0.65),
                (width, height)]
    # now we just require the lower triangular region if the image
    cropped_image = region_of_interest(canny_image, np.array([region_of_interest_vertices], np.int32))

    # using line detection algorithm
    lines = cv2.HoughLinesP(cropped_image, rho=2, theta=np.pi/180, threshold=50, lines=np.array([]),
                            maxLineGap=150, minLineLength=40)

    # Drawing the line on image
    image_with_line = draw_the_lines(image, lines)

    return image_with_line


# for opening the video.
# video is being shown image by image.
video = cv2.VideoCapture('lane_detection_video.mp4')


while video.isOpened():
    # is_grabbed is going to return a boolean value weather the frame was returned successfully or not.
    is_grabbed, frame = video.read()

    # if the boolean value of is_grabbed is false
    # it will enter 'if not' if the statement ahead would be false
    # 'is_grabbed' will become false when the frame won't be returning any value i.e. end of the video
    if not is_grabbed:
        break
    # for making the frames gray
    frame = get_lane_detected(frame)
    cv2.imshow('lane detection video', frame)
    cv2.waitKey(15)


video.release()
cv2.destroyAllWindows()

