import random

import cv2 as cv
import numpy as np
import math
import copy

from CV_Final_Project_Musical_Character_Recognition.python_files.data_types.Accidental import Accidental
from CV_Final_Project_Musical_Character_Recognition.python_files.data_types.Beam import Beam
from CV_Final_Project_Musical_Character_Recognition.python_files.data_types.Note import Note
from CV_Final_Project_Musical_Character_Recognition.python_files.data_types.Rest import Rest
from CV_Final_Project_Musical_Character_Recognition.python_files.image_manipulation import cleanup_image
from CV_Final_Project_Musical_Character_Recognition.python_files.processing_utils import get_distance, \
    find_best_match_for_image
from CV_Final_Project_Musical_Character_Recognition.python_files import processing_utils

# CODE SNIPPET TAKEN FROM https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
# THIS IS NOT MY CODE, applies sigma parameter to auto create lower and upper bounds of Canny edge detector and returns
# the edge img result
def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv.Canny(image, lower, upper)
    # return the edged image
    return edged


def identify_horizontal_lines(gray_sheet_music, hough_thresh, hough_min_line_length, hough_max_line_gap,
                              show_edges):
    temp_img = copy.deepcopy(gray_sheet_music)
    # perform opening to remove majority of horizontal noise
    horizontal_noise_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 1))
    temp_img = 255 - cv.morphologyEx(255 - temp_img, cv.MORPH_OPEN, horizontal_noise_kernel, iterations=1)

    # perform opening to remove majority of horizontal noise
    #vertical_noise_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 4))
    #temp_img = 255 - cv.morphologyEx(255 - temp_img, cv.MORPH_CLOSE, vertical_noise_kernel, iterations=1)

    # create edge image from temp_img
    edges_binary_sheet_music = auto_canny(temp_img, 0.5)

    # show img if requested
    if show_edges:
        cv.imshow("sheet_music_edges", edges_binary_sheet_music)
        cv.waitKey()

    # detect all horizontal lines
    horizontal_lines = cv.HoughLinesP(edges_binary_sheet_music, rho=1, theta=math.pi / 180, threshold=hough_thresh,
                                      minLineLength=hough_min_line_length, maxLineGap=hough_max_line_gap)

    return horizontal_lines


def filter_and_sort_horizontal_lines(horizontal_lines, merge_threshold):
    # sort horizontal lines by y value
    horizontal_lines_sorted = sorted(horizontal_lines, key=lambda x: x[0][1])

    # list of unneeded lines
    indices_to_be_deleted = []
    # num of items to delete
    count = 0
    min_x = 1000000
    max_x = -1
    for i in range(1, len(horizontal_lines_sorted)):
        prev_x1, prev_y1, prev_x2, prev_y2 = horizontal_lines_sorted[i - 1][0]
        x1, y1, x2, y2 = horizontal_lines_sorted[i][0]

        # if previous line is closer than merge_threshold to y1, delete y1 and merge lines by averaging
        if abs(y1 - prev_y1) <= merge_threshold:
            indices_to_be_deleted.append(i - count)
            horizontal_lines_sorted[i - 1][0][1] = horizontal_lines_sorted[i - 1][0][3] = ((prev_y1 + y1) / 2)
            count += 1
            continue

    # delete unnecessary lines
    for i in indices_to_be_deleted:
        horizontal_lines_sorted.pop(i)

    return horizontal_lines_sorted


def create_staves(sheet_music, horizontal_lines_sorted, spacing):
    sheet_music_new = copy.deepcopy(sheet_music)

    # identify groups of staves based on spacing
    staves = []
    # current stave group
    curr_group = list(horizontal_lines_sorted[0])
    for i in range(1, len(horizontal_lines_sorted)):
        # grab curr and prev y values of lines to compare
        y_prev_line = horizontal_lines_sorted[i - 1][0][1]
        y_curr_line = horizontal_lines_sorted[i][0][1]
        # if distance between y's is less than spacing add line to current group
        if abs(y_prev_line - y_curr_line) < 30:
            curr_group.append(horizontal_lines_sorted[i][0])
        # else we are done with current group, append curr_group to groups and remake curr_group
        else:
            staves.append(curr_group)
            curr_group = list(horizontal_lines_sorted[i])
    # append final group to groups
    staves.append(curr_group)

    # if line group does not have 5 lines then it is not a stave, remove it
    removal_count = 0
    for i in range(0, len(staves)):
        if len(staves[i - removal_count]) < 5:
            staves.pop(i - removal_count)
            removal_count += 1

    print(removal_count)
    # determine stave spacing
    stave_spacing = abs(staves[0][0][1] - staves[0][1][1])
    print(stave_spacing)

    for stave in staves:
        min_x = 1000000
        max_x = -1
        for line in stave:
            x1, _, x2, _ = line
            if x1 < min_x:
                min_x = x1
            if x2 > max_x:
                max_x = x2

        for line in stave:
            line[0] = min_x
            line[2] = max_x

    # add in two additional lines into each group above 1st and below 5th using stave spacing
    for group in staves:
        group.insert(0, [group[0][0], group[0][1] - stave_spacing, group[0][2], group[0][3] - stave_spacing])
        group.insert(0, [group[0][0], group[0][1] - stave_spacing, group[0][2], group[0][3] - stave_spacing])
        group.insert(len(group), [group[len(group) - 1][0], group[len(group) - 1][1] + stave_spacing,
                                  group[len(group) - 1][2], group[len(group) - 1][3] + stave_spacing])
        group.insert(len(group), [group[len(group) - 1][0], group[len(group) - 1][1] + stave_spacing,
                                  group[len(group) - 1][2], group[len(group) - 1][3] + stave_spacing])

    return staves, stave_spacing


def identify_clef(sheet_music, stave, w, show_sift):
    x1 = stave[0][0]
    y1 = stave[0][1]
    x2 = stave[0][0] + w
    y2 = stave[len(stave) - 1][1]

    clef_img = sheet_music[y1:y2, x1:x2]

    templates = {}
    # bass clef
    bass_clef = cv.imread("./clef_types/bass_clef.jpg", 0)
    templates['bass_clef'] = bass_clef
    # treble clef
    treble_clef = cv.imread("./clef_types/treble_clef.jpg", 0)
    templates['treble_clef'] = treble_clef

    clef_name = find_best_match_for_image(clef_img, templates, "clef", show_sift)
    return clef_name


def identify_accidental(sheet_music, stave, clef, clef_width, accidental_width, on_line_threshold,
                        stave_padding, show_sift_accidental):
    if accidental_width == 0:
        return []

    all_accidentals = []

    x1 = stave[0][0] + clef_width
    y1 = stave[0][1]
    x2 = stave[0][0] + clef_width + accidental_width
    y2 = stave[len(stave) - 1][1]

    templates = {}
    # sharp accidental
    sharp = cv.imread("./accidental_types/sharp.png", 0)
    templates['sharp'] = sharp
    # flat accidental
    flat = cv.imread("./accidental_types/flat.png", 0)
    templates['flat'] = flat
    # natural accidental
    natural = cv.imread("./accidental_types/natural.png", 0)
    templates['natural'] = natural

    all_accidentals_img = sheet_music[y1:y2, x1:x2]
    accidentals_img, accidentals_centroids = identify_accidentals_as_ccs(sheet_music,
                                                                         cv.cvtColor(all_accidentals_img,
                                                                                     cv.COLOR_BGR2GRAY), 1)
    for centroid in accidentals_centroids:
        centroid[0] += x1
        centroid[1] += y1

        accidental_img = sheet_music[centroid[1] - 10:centroid[1] + 10,
                         centroid[0] - 10:centroid[0] + 10]

        accidental_name = find_best_match_for_image(accidental_img, templates, "accidental", show_sift_accidental)
        accidental_note = point_to_note([], centroid, 30, stave, clef, on_line_threshold, stave_padding)

        all_accidentals.append(Accidental(accidental_note.name, accidental_name, centroid, 30))

    return all_accidentals


def identify_potential_notes(binary_image, sheet_music, dist_threshold, show_all, img_details_size):
    sheet_music_new = copy.deepcopy(sheet_music)

    binary_image = copy.deepcopy(binary_image)
    if img_details_size == "small":
        binary_image = 255 - cv.erode(binary_image, np.ones((1, 1)))
        binary_image = 255 - cv.dilate(binary_image, np.ones((1, 1)))
        binary_image = cv.erode(binary_image, np.ones((2, 2)))
    elif img_details_size == "medium":
        binary_image = 255 - cv.erode(binary_image, np.ones((2, 2)))
        binary_image = 255 - cv.dilate(binary_image, np.ones((3, 3)))
        binary_image = cv.erode(binary_image, np.ones((2, 2)))
    elif img_details_size == "large":
        binary_image = 255 - cv.erode(binary_image, np.ones((2, 2)))
        binary_image = 255 - cv.dilate(binary_image, np.ones((4, 4)))
        binary_image = cv.erode(binary_image, np.ones((3, 3)))

    #binary_image = cleanup_image(cv.cvtColor(binary_image, cv.COLOR_BGR2GRAY), (1, 1), (1, 1))

    #if show_all:
    cv.imshow("eroded image", binary_image)
    cv.waitKey()

    # Setup SimpleBlobDetector parameters.
    params = cv.SimpleBlobDetector_Params()
    #params.minThreshold = 1
    #params.maxThreshold = 300
    # - Filter by Area
    params.filterByArea = True
    if img_details_size == "small":
        params.minArea = 10
    elif img_details_size == "medium":
        params.minArea = 60
    elif img_details_size == "large":
        params.minArea = 80
    # - Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.85

    # Create a detector with the parameters
    detector = cv.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints = detector.detect(binary_image)
    print([i.pt for i in keypoints])

    # delete unnecessary blobs
    for keypoint1 in keypoints:
        for keypoint2 in keypoints:
            if keypoint1 == keypoint2:
                continue
            if get_distance(keypoint1.pt, keypoint2.pt) <= dist_threshold:
                keypoints.remove(keypoint2)

    return keypoints


def identify_time_signature(sheet_music, stave, clef_width, accidental_width, time_sig_width, show_sift):
    x1 = stave[0][0] + clef_width + accidental_width
    y1 = stave[0][1]
    x2 = stave[0][0] + clef_width + accidental_width + time_sig_width
    y2 = stave[len(stave) - 1][1]

    templates = {}
    # 2_4 time
    two_four = cv.imread("./time_sig_types/2_4.png", 0)
    templates['2_4'] = two_four
    # 3_4 time
    three_four = cv.imread("./time_sig_types/3_4.png", 0)
    templates['3_4'] = three_four
    # 4_4 time
    four_four = cv.imread("./time_sig_types/4_4.png", 0)
    templates['4_4'] = four_four
    # 6_8 time
    six_eight = cv.imread("./time_sig_types/6_8.png", 0)
    templates['6_8'] = six_eight

    time_sig_img = sheet_music[y1:y2, x1:x2]
    time_sig_name = find_best_match_for_image(time_sig_img, templates, "time_signature", show_sift)
    return time_sig_name


def points_to_notes(accidentals, points, stave, clef, clef_width, on_line_threshold, min_size_threshold, stave_padding):
    notes = []
    for point in points:
        if point.size >= min_size_threshold:
            note = point_to_note(accidentals, point.pt, point.size, stave, clef, clef_width, on_line_threshold, stave_padding)
            if note.octave != -1:
                notes.append(note)
    return notes


def point_to_note(accidentals, point, pt_size, stave, clef, clef_width, on_line_threshold, stave_padding):
    note_name = ""
    note_octave = -1

    if clef == 'treble_clef':
        treble = True
    else:
        treble = False

    if stave[0][1] - stave_padding <= point[1] <= (stave[len(stave) - 1][1] + stave_padding):
        if point[0] > stave[0][0] + clef_width:
            last_line = stave[0]
            last_j = 0
            for j in range(0, len(stave)):
                y_blob = point[1]
                y_line = stave[j][1]

                note_on_line = False

                # blob on line
                if abs(y_blob - y_line) <= on_line_threshold:
                    print("Note on this line:", j)
                    note_on_line = True
                    note_name, note_octave = processing_utils.calculate_note(j, note_on_line, treble)
                    note_accidental = "natural"
                    for accidental in accidentals:
                        if accidental.note == note_name:
                            note_accidental = accidental.type
                        else:
                            note_accidental = "natural"
                    return Note(note_name, note_octave, None, note_accidental, point, pt_size)

                # blob not on line
                if (last_line[1] <= y_blob <= y_line) and (note_on_line is False):
                    print("Note within these lines: (", last_j, j, ")")
                    note_name, note_octave = processing_utils.calculate_note(last_j, note_on_line, treble)
                    note_accidental = "natural"
                    for accidental in accidentals:
                        if accidental.note == note_name:
                            note_accidental = accidental.type
                        else:
                            note_accidental = "natural"
                    return Note(note_name, note_octave, None, note_accidental, point, pt_size)

                last_line = stave[j]
                last_j = j

    return Note(note_name, note_octave, None, "natural", point, pt_size)


def identify_note_lengths_for_stave(sheet_music, stave, w, h, show_sift):
    templates = {}
    # quarter note
    quarter_note_head = cv.imread("./length_types/quar_note_head.png", 0)
    templates['quarter_note'] = quarter_note_head
    # half note
    half_note_head = cv.imread("./length_types/half_note_head.png", 0)
    templates['half_note'] = half_note_head
    # full note
    full_note_head = cv.imread("./length_types/full_note_head.png", 0)
    templates['full_note'] = full_note_head

    for note in stave.notes:
        note_img = sheet_music[int(note.pt[1] - h / 2):int(note.pt[1] + h / 2),
                   int(note.pt[0] - w / 2):int(note.pt[0] + w / 2)]

        note_length_name = find_best_match_for_image(note_img, templates, "note_length", show_sift)
        if note_length_name is None:
            note_length_name = "unknown"

        beam_for_note_count = 0
        for beam in stave.beams:
            if beam.pt1[0] <= note.pt[0] <= beam.pt2[0]:
                beam_for_note_count += 1

        if beam_for_note_count == 1:
            note_length_name = "eighth_note"
        elif beam_for_note_count == 2:
            note_length_name = "sixteenth_note"

        note.length = note_length_name


def identify_accidentals_as_ccs(sheet_music, accidentals_img, dist_threshold):
    sheet_music_new = copy.deepcopy(sheet_music)

    # binary_image = cleanup_image(cv.cvtColor(accidentals_img, cv.COLOR_BGR2GRAY), (2, 2), (1, 1))
    # binary_image = 255 - cv.erode(binary_image, np.ones((3, 1)))

    # cv.imshow("eroded image for rests", binary_image)

    contours, hierarchy = cv.findContours(accidentals_img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    print(len(contours))
    print(contours, hierarchy)

    accidental_centroids = []
    for i in range(1, len(contours) - 1):
        area = cv.contourArea(contours[i])
        if 5 < area < 400:
            M = cv.moments(contours[i])
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            cv.drawContours(accidentals_img, contours, i, (0, 255, 0), 1)
            cv.drawMarker(accidentals_img, (cX, cY), (255, 0, 0), markerType=cv.MARKER_CROSS)
            accidental_centroids.append([cX, cY])
    # cv.drawContours(sheet_music_new, contours, -1, (0,255,0), 1)

    # cv.imshow("sheet music with accidentals", accidentals_img)
    # cv.waitKey()

    return sheet_music_new, accidental_centroids


def identify_beams_as_contours(sheet_music, stave, stave_padding, clef_width, accidental_width, time_sig_width, note_width):
    sheet_music_new = copy.deepcopy(sheet_music)

    sheet_music = cv.cvtColor(sheet_music, cv.COLOR_BGR2GRAY)
    # binary_image = 255 - cv.erode(binary_image, np.ones((3, 1)))

    # cv.imshow("eroded image for rests", binary_image)

    contours, hierarchy = cv.findContours(sheet_music, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    print(len(contours))
    print(contours, hierarchy)

    beams = []
    for i in range(1, len(contours) - 1):
        area = cv.contourArea(contours[i])
        x, y, w, h = cv.boundingRect(contours[i])
        if 5 < area < 1000:
            M = cv.moments(contours[i])
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            cv.drawMarker(sheet_music_new, (cX, cY), (255, 255, 0), markerType=cv.MARKER_CROSS)
            beams.append(Beam([int(x - note_width / 2), y], [int(x + w + note_width/2), y + h]))
    cv.drawContours(sheet_music_new, contours, -1, (0, 255, 0), 1)

    beams_in_stave = []
    for beam in beams:
        if stave[0][1] <= beam.pt1[1] <= (stave[len(stave) - 1][1]):
            if beam.pt1[0] >= (stave[0][0] + clef_width):
                beams_in_stave.append(beam)
                cv.rectangle(sheet_music_new, beam.pt1, beam.pt2, (255, 0, 0))

    #cv.imshow("sheet music with beams", sheet_music_new)
    #cv.waitKey()

    return sheet_music_new, beams_in_stave


def identify_rests_as_contours(sheet_music, sheet_music_clean, stave, stave_padding, clef_width, accidental_width,
                               time_sig_width, show_sift):
    sheet_music_new = copy.deepcopy(sheet_music)

    sheet_music_clean = copy.deepcopy(sheet_music_clean)
    sheet_music_clean = cv.cvtColor(sheet_music_clean, cv.COLOR_BGR2GRAY)
    # perform opening to create blobs for rests
    horizontal_noise_kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    sheet_music_clean = 255 - cv.morphologyEx(255 - sheet_music_clean, cv.MORPH_OPEN, horizontal_noise_kernel, iterations=1)
    # perform closing to create blobs for rests
    horizontal_noise_kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 7))
    sheet_music_clean = 255 - cv.morphologyEx(255 - sheet_music_clean, cv.MORPH_CLOSE, horizontal_noise_kernel, iterations=3)
    # binary_image = 255 - cv.erode(binary_image, np.ones((3, 1)))

    #cv.imshow("eroded image for rests", sheet_music_clean)
    #cv.waitKey()

    contours, hierarchy = cv.findContours(sheet_music_clean, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    print(contours, hierarchy)

    rests = []
    for i in range(1, len(contours) - 1):
        area = cv.contourArea(contours[i])
        if 15 < area < 200:
            M = cv.moments(contours[i])
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            rests.append(Rest([cX, cY], None))
    cv.drawContours(sheet_music_new, contours, -1, (0, 255, 0), 1)

    templates = {}
    # full_rest
    full_rest = cv.imread("./rest_types/full_rest.png", 0)
    templates['full_rest'] = full_rest
    # half_rest
    half_rest = cv.imread("./rest_types/half_rest.png", 0)
    templates['half_rest'] = half_rest
    # quarter_rest
    quarter_rest = cv.imread("./rest_types/quarter_rest.png", 0)
    templates['quarter_rest'] = quarter_rest
    # eighth_rest
    eighth_rest = cv.imread("./rest_types/eighth_rest.png", 0)
    templates['eighth_rest'] = eighth_rest
    # sixteenth_rest
    sixteenth_rest = cv.imread("./rest_types/sixteenth_rest.png", 0)
    templates['sixteenth_rest'] = sixteenth_rest


    rests_in_stave = []
    for rest in rests:
        if stave[0][1] - stave_padding <= rest.pt[1] <= (stave[len(stave) - 1][1] + stave_padding):
            if rest.pt[0] >= (stave[0][0] + clef_width + accidental_width + time_sig_width):
                rests_in_stave.append(rest)
                cv.drawMarker(sheet_music_new, rest.pt, (255, 0, 0), markerType=cv.MARKER_CROSS)

                rest_img = sheet_music[rest.pt[1] - 15:rest.pt[1] + 15,
                                       rest.pt[0] - 10:rest.pt[0] + 10]

                rest_name = find_best_match_for_image(rest_img, templates, "rest", show_sift)
                if rest_name is None:
                    rest_name = "unknown"
                rest.type = rest_name




    #cv.imshow("sheet music with rests", sheet_music_new)
    #cv.waitKey()

    return sheet_music_new, rests_in_stave
