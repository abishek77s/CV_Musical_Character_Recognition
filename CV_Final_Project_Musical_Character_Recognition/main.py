import time
import cv2 as cv
from playsound import playsound

from CV_Final_Project_Musical_Character_Recognition.python_files \
    import object_identification, image_manipulation, draw
from CV_Final_Project_Musical_Character_Recognition.python_files.data_types.Stave import Stave
from CV_Final_Project_Musical_Character_Recognition.python_files.image_manipulation import cleanup_image


def sort_notes(staves_populated):
    for stave in staves_populated:
        stave.notes = sorted(stave.notes, key=lambda x: x.pt[0])
    return staves_populated


def play_keys(staves_populated):
    for stave in staves_populated:
        for note in stave.notes:
            filename = r"mp3_notes/" + str(note.name) + str(note.octave) + ".mp3"
            print(filename)
            if note.length == 'sixteenth_note':
                playsound(filename)
            elif note.length == 'eighth_note':
                playsound(filename)
            elif note.length == 'quarter_note':
                playsound(filename)
            elif note.length == 'half_note':
                playsound(filename)
                playsound(filename)
            elif note.length == 'full_note':
                playsound(filename)
                playsound(filename)
                playsound(filename)
                playsound(filename)
            else:
                print("ERROR: Note", note.name, "has an unknown length type.")
                pass

            time.sleep(0.3)


def populate_staves(sheet_music, sheet_music_clean, staves, note_blobs, clef_width, accidental_width,
                    time_sig_width, stave_padding, on_line_threshold, min_size_threshold, note_width, note_height,
                    show_img_dict):
    staves_populated = []
    for stave in staves:
        staves_populated.append(populate_stave(sheet_music, sheet_music_clean, stave, note_blobs, clef_width,
                                               accidental_width, time_sig_width, stave_padding,
                                               on_line_threshold, min_size_threshold, note_width, note_height, show_img_dict))
    return staves_populated


def populate_stave(sheet_music, sheet_music_clean, stave, note_blobs, clef_width, accidental_width,
                   time_sig_width, stave_padding, on_line_threshold, min_size_threshold, note_width, note_height,
                   show_img_dict):

    stave_line_spacing = abs(stave[1][1] - stave[0][1])

    clef = object_identification.identify_clef(sheet_music, stave, clef_width, show_img_dict['show_sift_clef'])
    accidentals = object_identification.identify_accidental(sheet_music_clean, stave, clef, clef_width,
                                                            accidental_width, on_line_threshold, stave_padding,
                                                            show_img_dict['show_sift_accidental'])
    time_signature = object_identification.identify_time_signature(sheet_music, stave, clef_width, accidental_width,
                                                                   time_sig_width, show_img_dict['show_sift_time_sig'])
    notes = object_identification.points_to_notes(accidentals, note_blobs, stave, clef, clef_width, on_line_threshold,
                                                  min_size_threshold, stave_padding)

    sheet_music_with_stuff_highlighted, beams = \
        object_identification.identify_beams_as_contours(sheet_music_clean, stave, stave_padding, clef_width,
                                                         accidental_width, time_sig_width, note_width)

    stave_populated = Stave(stave, stave_line_spacing, clef, accidentals, time_signature, notes, beams, [])
    object_identification.identify_note_lengths_for_stave(sheet_music, stave_populated, note_width, note_height,
                                                          show_img_dict['show_sift_notes'])

    return stave_populated


def print_stave_data(staves_populated):
    for stave in staves_populated:
        stave.to_str()
        for accidental in stave.accidentals:
            accidental.to_str()
        for note in stave.notes:
            note.to_str()
        for beam in stave.beams:
            beam.to_str()


if __name__ == "__main__":
    # ---------------------------- Handle initial image grab and grayscale conversion ---------------------------------
    sheet_music = cv.imread("music_sheets/Make_You_Feel_My_Love.png")  # read in sheet music image
    sheet_music_resized = cv.resize(sheet_music, (sheet_music.shape[1] // 2, sheet_music.shape[0] // 2))
    gray_sheet_music = cv.cvtColor(sheet_music, cv.COLOR_BGR2GRAY)  # convert image to grayscale
    # sheet_music = image_manipulation.cleanup_image(gray_sheet_music, (1, 1), (1, 1))
    # -----------------------------------------------------------------------------------------------------------------

    # ============================================== All Variables =====================================================
    #

    img_details_size = "medium"  # small, medium, or large
    has_half_notes = False  # Check if we should perform additional opening and closings to image to detect half notes

    clef_width = 45  # distance from staves group start to end of clef
    accidental_width = 0
    time_sig_width = 30

    resize_img = True
    dim = (sheet_music.shape[1] // 2, sheet_music.shape[0] // 2)

    # ----------------- handle which images to show during processing -----------------
    show_img_dict = {
        'show_all': False,
        'show_blobs': True,
        'show_informational': True,
        'show_morph': True,
        'show_edges': False,
        'show_sift_clef': False,
        'show_sift_notes': False,
        'show_sift_accidental': False,
        'show_sift_time_sig': False,
        'show_sift_rests': False
    }
    # ---------------------------------------------------------------------------------
    if img_details_size == "small":
        min_blob_size_threshold = 7  # min size required to process blob as a note
        lines_horiz_ksize = 2
        lines_vert_ksize = 2
        on_line_threshold = 2  # threshold for assigning blob to being on a line in stave
        note_width = 20
        note_height = 10
    elif img_details_size == "medium":
        min_blob_size_threshold = 14  # min size required to process blob as a note
        lines_horiz_ksize = 3
        lines_vert_ksize = 3
        on_line_threshold = 3  # threshold for assigning blob to being on a line in stave
        note_width = 20
        note_height = 15
    elif img_details_size == "large":
        min_blob_size_threshold = 13  # min size required to process blob as a note
        lines_horiz_ksize = 4
        lines_vert_ksize = 3
        on_line_threshold = 5  # threshold for assigning blob to being on a line in stave
        note_width = 40
        note_height = 30
    # ---------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------
    #
    # ----------------------- Manage stave line detection -----------------------------
    # - Hough transform variables for horizontal line detection -
    staves_hough_thresh = 300
    staves_hough_min_line_length = 200
    staves_hough_max_line_gap = 50
    # - handle the creation of stave groups -
    line_merge_threshold = 2
    max_stave_line_spacing = 20
    # ---------------------------------------------------------------------------------
    #
    # =================================================================================================================

    # ------------------ Identify horizontal lines in sheet music for later use in stave creation ---------------------
    # --- Find all viable horizontal lines
    horizontal_lines = object_identification.identify_horizontal_lines(gray_sheet_music, staves_hough_thresh,
                                                                       staves_hough_min_line_length,
                                                                       staves_hough_max_line_gap,
                                                                       show_img_dict['show_edges'])
    # --- Filter and sort viable horizontal lines
    horizontal_lines_sorted = object_identification.filter_and_sort_horizontal_lines(horizontal_lines,
                                                                                     line_merge_threshold)
    # -----------------------------------------------------------------------------------------------------------------

    # ---------------------- Identify stave lines and positions for later use in note definition ----------------------
    staves, stave_spacing = \
        object_identification.create_staves(sheet_music, horizontal_lines_sorted, max_stave_line_spacing)
    # -----------------------------------------------------------------------------------------------------------------

    # --------------- process image to remove horizontal lines, and create highlight of lines removed -----------------

    sheet_music_no_lines = image_manipulation.remove_lines(sheet_music, lines_horiz_ksize, lines_vert_ksize,
                                                           show_img_dict['show_morph'])

    #binary_img = cleanup_image(cv.cvtColor(sheet_music_no_lines, cv.COLOR_BGR2GRAY), (1, 1), (1, 1))
    sheet_music_no_lines_half_notes_filled = image_manipulation.fill_in_half_notes(sheet_music_no_lines,
                                                                                   has_half_notes, img_details_size,
                                                                                   show_img_dict['show_morph'])

    # -----------------------------------------------------------------------------------------------------------------

    # ----------------------------- identify potential notes from blob detection --------------------------------------
    note_blobs = object_identification.identify_potential_notes(sheet_music_no_lines_half_notes_filled, sheet_music, 1,
                                                                show_img_dict['show_all'], img_details_size)
    # -----------------------------------------------------------------------------------------------------------------

    sheet_music_no_lines_no_notes = image_manipulation.remove_note_heads(sheet_music_no_lines, note_blobs,
                                                                         note_width, note_height,
                                                                         min_blob_size_threshold)

    # -----------------------------------------------------------------------------------------------------------------

    staves_populated = populate_staves(sheet_music, sheet_music_no_lines_no_notes, staves, note_blobs, clef_width,
                                       accidental_width, time_sig_width, stave_spacing * 2, on_line_threshold,
                                       min_blob_size_threshold, note_width, note_height, show_img_dict)

    sheet_music_no_lines_no_notes_no_intro = image_manipulation.remove_intro_data(sheet_music_no_lines_no_notes,
                                                                                  staves_populated, clef_width,
                                                                                  accidental_width, time_sig_width)
    sheet_music_no_lines_no_notes_no_intro_no_beams = \
        image_manipulation.remove_beams(sheet_music_no_lines_no_notes_no_intro, staves_populated)

    cv.imshow("sheet music clean", sheet_music_no_lines_no_notes_no_intro_no_beams)
    cv.waitKey()

    for stave in staves_populated:
        sheet_music_with_more_stuff_highlighted, stave.rests = \
            object_identification.identify_rests_as_contours(sheet_music, sheet_music_no_lines_no_notes_no_intro_no_beams,
                                                             stave.lines, stave.line_spacing, clef_width, accidental_width,
                                                             time_sig_width, show_img_dict['show_sift_rests'])

    if show_img_dict['show_all']:
        cv.imshow("original_img", sheet_music)
        cv.imshow("original_img_gray", gray_sheet_music)
        cv.waitKey()
        
    if show_img_dict['show_all']:
        cv.imshow("sheet_music_no_horizontal_lines", sheet_music_no_lines)
        cv.waitKey()

    if show_img_dict['show_all']:
        cv.imshow("sheet_music_no_lines_half_notes_filled", sheet_music_no_lines_half_notes_filled)
        cv.waitKey()
        
    if show_img_dict['show_informational']:
        sheet_music_with_staves = draw.draw_horizontal_lines(sheet_music, staves_populated, resize_img, dim)
        cv.imshow("sheet music with staves", sheet_music_with_staves)
        cv.waitKey()
        
    if show_img_dict['show_blobs']:
        sheet_music_with_potential_notes = draw.draw_blobs(sheet_music, note_blobs, resize_img, dim)
        cv.imshow("sheet music with potential notes highlighted", sheet_music_with_potential_notes)
        cv.waitKey()
        
    if show_img_dict['show_informational']:
        sheet_music_with_clefs = draw.draw_clef(sheet_music, staves_populated, clef_width, resize_img, dim)
        cv.imshow("sheet music with clefs highlighted", sheet_music_with_clefs)
        cv.waitKey()

    if show_img_dict['show_informational']:
        sheet_music_with_accidentals = draw.draw_accidentals(sheet_music, staves_populated, resize_img, dim)
        cv.imshow("sheet music with accidentals highlighted", sheet_music_with_accidentals)
        cv.waitKey()

    if show_img_dict['show_informational']:
        sheet_music_with_time_signature = draw.draw_time_signature(sheet_music, staves_populated, clef_width,
                                                                   accidental_width, time_sig_width, resize_img, dim)
        cv.imshow("sheet music with time signature highlighted", sheet_music_with_time_signature)
        cv.waitKey()
    
    if show_img_dict['show_informational']:
        sheet_music_updated_with_notes = draw.draw_notes(sheet_music, staves_populated, resize_img, dim)
        cv.imshow("sheet music with notes", sheet_music_updated_with_notes)
        cv.waitKey()

    if show_img_dict['show_all']:
        cv.imshow("sheet music no lines no notes", sheet_music_no_lines_no_notes)
        cv.waitKey()
        
    if show_img_dict['show_informational']:
        sheet_music_updated_with_note_lengths = draw.draw_note_lengths(sheet_music, staves_populated, resize_img, dim)
        cv.imshow("sheet music with note lengths", sheet_music_updated_with_note_lengths)
        cv.waitKey()

    if show_img_dict['show_informational']:
        sheet_music_with_rests = draw.draw_rests(sheet_music, staves_populated, resize_img, dim)
        cv.imshow("sheet music with rests", sheet_music_with_rests)
        cv.waitKey()

    print_stave_data(staves_populated)

    play_keys(sort_notes(staves_populated))
