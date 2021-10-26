import cv2 as cv


# Mouse callback function. Appends the x,y location of mouse click to a list.
def get_xy(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        print(x, y)
        param.append((x, y))
        #cv.destroyWindow("building_config")


def main(img_path):
    print("hello")
    img = cv.imread(img_path)

    param = []
    cv.namedWindow("Viewer")
    cv.setMouseCallback("Viewer", get_xy, param)
    cv.imshow("Viewer", img)
    cv.waitKey()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main("Notes_and_Rests.png")
