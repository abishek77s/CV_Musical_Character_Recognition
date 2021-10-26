import cv2 as cv
import copy


def perform_sift_on_time(bgr_image1, bgr_image2, gray_image1, gray_image2, show_sift):
    matcher = cv.BFMatcher(cv.NORM_L2)

    detector = cv.xfeatures2d.SIFT_create(
        nfeatures=100,
        nOctaveLayers=3,  # default = 3
        contrastThreshold=0.04,  # default = 0.04
        edgeThreshold=10,  # default = 10
        sigma=1.6  # default = 1.6
    )

    kp1, desc1 = detector.detectAndCompute(gray_image1, mask=None)
    bgr_display1 = copy.deepcopy(bgr_image1)
    cv.drawKeypoints(image=bgr_display1, keypoints=kp1, outImage=bgr_display1,
                     flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    detector = cv.xfeatures2d.SIFT_create(
        nfeatures=100,
        nOctaveLayers=3,  # default = 3
        contrastThreshold=0.04,  # default = 0.04
        edgeThreshold=10,  # default = 10
        sigma=1.6  # default = 1.6
    )
    kp2, desc2 = detector.detectAndCompute(gray_image2, mask=None)
    bgr_display2 = copy.deepcopy(bgr_image2)
    cv.drawKeypoints(image=bgr_display2, keypoints=kp2, outImage=bgr_display2,
                     flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Use k nearest neighbor matching and apply ratio test.
    matches = matcher.knnMatch(desc1, desc2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)
    matches = good
    print("Number of matches between images: ", len(matches))

    bgr_matches = cv.drawMatches(img1=bgr_image1, keypoints1=kp1, img2=bgr_image2, keypoints2=kp2, matches1to2=matches,
                                 matchesMask=None, outImg=None)

    if show_sift:
        cv.imshow("matches", bgr_matches)
        cv.waitKey()

    return len(matches)


def perform_sift_on_clef(bgr_image1, bgr_image2, gray_image1, gray_image2, show_sift):
    matcher = cv.BFMatcher(cv.NORM_L2)

    detector = cv.xfeatures2d.SIFT_create(
        nfeatures=100,
        nOctaveLayers=3,  # default = 3
        contrastThreshold=0.004,  # default = 0.04
        edgeThreshold=0.001,  # default = 10
        sigma=0.8  # default = 1.6
    )

    kp1, desc1 = detector.detectAndCompute(gray_image1, mask=None)
    bgr_display1 = copy.deepcopy(bgr_image1)
    cv.drawKeypoints(image=bgr_display1, keypoints=kp1, outImage=bgr_display1,
                     flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    detector = cv.xfeatures2d.SIFT_create(
        nfeatures=100,
        nOctaveLayers=3,  # default = 3
        contrastThreshold=0.04,  # default = 0.04
        edgeThreshold=10,  # default = 10
        sigma=1.6  # default = 1.6
    )
    kp2, desc2 = detector.detectAndCompute(gray_image2, mask=None)
    bgr_display2 = copy.deepcopy(bgr_image2)
    cv.drawKeypoints(image=bgr_display2, keypoints=kp2, outImage=bgr_display2,
                     flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Use k nearest neighbor matching and apply ratio test.
    matches = matcher.knnMatch(desc1, desc2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)
    matches = good
    print("Number of matches between images: ", len(matches))

    bgr_matches = cv.drawMatches(img1=bgr_image1, keypoints1=kp1, img2=bgr_image2, keypoints2=kp2, matches1to2=matches,
                                 matchesMask=None, outImg=None)

    if show_sift:
        cv.imshow("matches", bgr_matches)
        cv.waitKey()

    return len(matches)


def perform_sift_on_head(bgr_image1, bgr_image2, gray_image1, gray_image2, show_sift):
    matcher = cv.BFMatcher(cv.NORM_L2)

    detector = cv.xfeatures2d.SIFT_create(
        nfeatures=100,
        nOctaveLayers=2,  # default = 3
        contrastThreshold=0.04,  # default = 0.04
        edgeThreshold=30,  # default = 10
        sigma=0.9   # default = 1.6
    )

    kp1, desc1 = detector.detectAndCompute(gray_image1, mask=None)
    bgr_display1 = copy.deepcopy(bgr_image1)
    cv.drawKeypoints(image=bgr_display1, keypoints=kp1, outImage=bgr_display1,
                     flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    detector = cv.xfeatures2d.SIFT_create(
        nfeatures=100,
        nOctaveLayers=2,  # default = 3
        contrastThreshold=0.04,  # default = 0.04
        edgeThreshold=30,  # default = 10
        sigma=0.9   # default = 1.6
    )
    kp2, desc2 = detector.detectAndCompute(gray_image2, mask=None)
    bgr_display2 = copy.deepcopy(bgr_image2)
    cv.drawKeypoints(image=bgr_display2, keypoints=kp2, outImage=bgr_display2,
                     flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Use k nearest neighbor matching and apply ratio test.
    matches = matcher.knnMatch(desc1, desc2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)
    matches = good
    print("Number of matches between images: ", len(matches))

    bgr_matches = cv.drawMatches(img1=bgr_image1, keypoints1=kp1, img2=bgr_image2, keypoints2=kp2, matches1to2=matches,
                                 matchesMask=None, outImg=None)

    if show_sift:
        cv.imshow("matches", bgr_matches)
        cv.waitKey()

    return len(matches)


def perform_sift_on_accidental(bgr_image1, bgr_image2, gray_image1, gray_image2, show_sift):
    matcher = cv.BFMatcher(cv.NORM_L2)

    detector = cv.xfeatures2d.SIFT_create(
        nfeatures=15,
        nOctaveLayers=3,  # default = 3
        contrastThreshold=0.04,  # default = 0.04
        edgeThreshold=10,  # default = 10
        sigma=0.8   # default = 1.6
    )

    kp1, desc1 = detector.detectAndCompute(gray_image1, mask=None)
    bgr_display1 = copy.deepcopy(bgr_image1)
    cv.drawKeypoints(image=bgr_display1, keypoints=kp1, outImage=bgr_display1,
                     flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    detector = cv.xfeatures2d.SIFT_create(
        nfeatures=15,
        nOctaveLayers=3,  # default = 3
        contrastThreshold=0.04,  # default = 0.04
        edgeThreshold=10,  # default = 10
        sigma=1   # default = 1.6
    )
    kp2, desc2 = detector.detectAndCompute(gray_image2, mask=None)
    bgr_display2 = copy.deepcopy(bgr_image2)
    cv.drawKeypoints(image=bgr_display2, keypoints=kp2, outImage=bgr_display2,
                     flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Use k nearest neighbor matching and apply ratio test.
    matches = matcher.knnMatch(desc1, desc2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)
    matches = good
    print("Number of matches between images: ", len(matches))

    bgr_matches = cv.drawMatches(img1=bgr_image1, keypoints1=kp1, img2=bgr_image2, keypoints2=kp2, matches1to2=matches,
                                 matchesMask=None, outImg=None)

    if show_sift:
        cv.imshow("matches", bgr_matches)
        cv.waitKey()

    return len(matches)


def perform_sift_on_rest(bgr_image1, bgr_image2, gray_image1, gray_image2, show_sift):
    matcher = cv.BFMatcher(cv.NORM_L2)

    detector = cv.xfeatures2d.SIFT_create(
        nfeatures=30,
        nOctaveLayers=3,  # default = 3
        contrastThreshold=0.04,  # default = 0.04
        edgeThreshold=10,  # default = 10
        sigma=0.85  # default = 1.6
    )

    kp1, desc1 = detector.detectAndCompute(gray_image1, mask=None)
    bgr_display1 = copy.deepcopy(bgr_image1)
    cv.drawKeypoints(image=bgr_display1, keypoints=kp1, outImage=bgr_display1,
                     flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    detector = cv.xfeatures2d.SIFT_create(
        nfeatures=30,
        nOctaveLayers=3,  # default = 3
        contrastThreshold=0.04,  # default = 0.04
        edgeThreshold=10,  # default = 10
        sigma=1  # default = 1.6
    )
    kp2, desc2 = detector.detectAndCompute(gray_image2, mask=None)
    bgr_display2 = copy.deepcopy(bgr_image2)
    cv.drawKeypoints(image=bgr_display2, keypoints=kp2, outImage=bgr_display2,
                     flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Use k nearest neighbor matching and apply ratio test.
    matches = matcher.knnMatch(desc1, desc2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)
    matches = good
    print("Number of matches between images: ", len(matches))

    bgr_matches = cv.drawMatches(img1=bgr_image1, keypoints1=kp1, img2=bgr_image2, keypoints2=kp2, matches1to2=matches,
                                 matchesMask=None, outImg=None)

    if show_sift:
        bgr_matches_resized = cv.resize(bgr_matches, (bgr_matches.shape[1]*4, bgr_matches.shape[0]*4))
        cv.imshow("matches", bgr_matches_resized)
        cv.waitKey()

    return len(matches)
