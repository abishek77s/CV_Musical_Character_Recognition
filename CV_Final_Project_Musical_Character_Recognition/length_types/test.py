import cv2 as cv
# import imutils
import glob, os
import numpy as np
import time


def create_template_image(filename, new_filename, x1, y1, x2, y2):
    image = cv.imread(filename)

    cv.imshow("showing", image[y1:y2, x1:x2])
    cv.waitKey()

    cv.imwrite(new_filename, image[y1:y2, x1:x2])


def find_fill_contours(gray_sheet_music):
    gray_sheet_music_new = gray_sheet_music.copy()

    fc = cv.findContours(gray_sheet_music_new, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = fc[0] if len(fc) == 2 else fc[1]

    closed_contours = []
    for contour in contours:
        if cv.contourArea(contour) > cv.arcLength(contour, True):
            print("Found closed contour")
            closed_contours.append(contour)
        else:
            print("Not a closed contour")

    for contour in contours:
        cv.drawContours(gray_sheet_music_new, [contour], 0, (255, 255, 255), -1)

    cv.imshow("contoured_image", gray_sheet_music_new)
    cv.waitKey()


if __name__ == "__main__":
    create_template_image("Notes_and_Rests.png", "../rest_types/sixteenth_rest.png", 643, 120, 673, 165)
