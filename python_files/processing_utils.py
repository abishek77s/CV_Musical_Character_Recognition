import math
import cv2 as cv

from CV_Final_Project_Musical_Character_Recognition.python_files import sift_calculations


def get_distance(pt1, pt2):
    return math.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)


def find_best_match_for_image(image, templates, sift_type, show_sift):
    #image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    best_match_name = ""
    best_num_matches = -1
    for template_name, template in templates.items():
        if sift_type == "clef":
            curr_num_matches = sift_calculations.perform_sift_on_clef(image, template, image, template, show_sift)
        elif sift_type == "accidental":
            curr_num_matches = sift_calculations.perform_sift_on_accidental(image, template, image, template, show_sift)
        elif sift_type == "time_signature":
            curr_num_matches = sift_calculations.perform_sift_on_time(image, template, image, template, show_sift)
        elif sift_type == "note_length":
            curr_num_matches = sift_calculations.perform_sift_on_head(image, template, image, template, show_sift)
        elif sift_type == "rest":
            curr_num_matches = sift_calculations.perform_sift_on_rest(image, template, image, template, show_sift)
        else:
            print("Please use correct sift type")
            curr_num_matches = -1

        if curr_num_matches >= best_num_matches:
            best_num_matches = curr_num_matches
            best_match_name = template_name

    if best_num_matches < 2:
        return None
    else:
        return best_match_name


def calculate_note(line_1, on_line, treble):
    note_name = ""
    note_octave = -1
    if treble:
        if on_line:
            if line_1 == 8:  # a3
                note_name = "a"
                note_octave = 3
            elif line_1 == 7:  # c4
                note_name = "c"
                note_octave = 4
            elif line_1 == 6:  # e4
                note_name = "e"
                note_octave = 4
            elif line_1 == 5:  # g4
                note_name = "g"
                note_octave = 4
            elif line_1 == 4:  # b4
                note_name = "b"
                note_octave = 4
            elif line_1 == 3:  # d5
                note_name = "d"
                note_octave = 5
            elif line_1 == 2:  # f5
                note_name = "f"
                note_octave = 5
            elif line_1 == 1:  # a5
                note_name = "a"
                note_octave = 5
            elif line_1 == 0:  # c6
                note_name = "c"
                note_octave = 6
        else:
            if line_1 == 7:  # b3
                note_name = "b"
                note_octave = 3
            elif line_1 == 6:  # d4
                note_name = "d"
                note_octave = 4
            elif line_1 == 5:  # f4
                note_name = "f"
                note_octave = 4
            elif line_1 == 4:  # a4
                note_name = "a"
                note_octave = 4
            elif line_1 == 3:  # c5
                note_name = "c"
                note_octave = 5
            elif line_1 == 2:  # e5
                note_name = "e"
                note_octave = 5
            elif line_1 == 1:  # g5
                note_name = "g"
                note_octave = 5
            elif line_1 == 0:  # b5
                note_name = "b"
                note_octave = 5
    else:
        if on_line:
            if line_1 == 8:  # c2
                note_name = "c"
                note_octave = 2
            elif line_1 == 7:  # e2
                note_name = "e"
                note_octave = 2
            elif line_1 == 6:  # g2
                note_name = "g"
                note_octave = 2
            elif line_1 == 5:  # b2
                note_name = "b"
                note_octave = 2
            elif line_1 == 4:  # d3
                note_name = "d"
                note_octave = 3
            elif line_1 == 3:  # f3
                note_name = "f"
                note_octave = 3
            elif line_1 == 2:  # a3
                note_name = "a"
                note_octave = 3
            elif line_1 == 1:  # c4
                note_name = "c"
                note_octave = 4
            elif line_1 == 0:  # e4
                note_name = "e"
                note_octave = 4
        else:
            if line_1 == 7:  # d2
                note_name = "b"
                note_octave = 5
            elif line_1 == 6:  # f2
                note_name = "b"
                note_octave = 5
            elif line_1 == 5:  # a2
                note_name = "b"
                note_octave = 5
            elif line_1 == 4:  # c3
                note_name = "b"
                note_octave = 5
            elif line_1 == 3:  # e3
                note_name = "b"
                note_octave = 5
            elif line_1 == 2:  # g3
                note_name = "b"
                note_octave = 5
            elif line_1 == 1:  # b3
                note_name = "b"
                note_octave = 5
            elif line_1 == 0:  # d4
                note_name = "b"
                note_octave = 5

    return note_name, note_octave
