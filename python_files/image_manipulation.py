import cv2 as cv
import copy


def cleanup_image(gray_sheet_music, opening_kernel, closing_kernel):

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, opening_kernel)
    binary_img = cv.morphologyEx(gray_sheet_music, cv.MORPH_OPEN, kernel)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, closing_kernel)
    binary_img = cv.morphologyEx(binary_img, cv.MORPH_CLOSE, kernel)

    thresh, binary_img = cv.threshold(binary_img, thresh=0, maxval=255, type=cv.THRESH_BINARY + cv.THRESH_OTSU)

    return binary_img


def remove_note_heads(sheet_music_no_horizontal_lines, note_blob_pts, w, h, size_threshold):
    sheet_music_no_horizontal_lines_no_notes = copy.deepcopy(sheet_music_no_horizontal_lines)

    for note in note_blob_pts:
        if note.size >= size_threshold:
            sheet_music_no_horizontal_lines_no_notes = \
                cv.rectangle(sheet_music_no_horizontal_lines_no_notes,
                             [int(note.pt[0] - w / 2), int(note.pt[1] - h / 2)],
                             [int(note.pt[0] + w / 2), int(note.pt[1] + h / 2)],
                             color=(255, 255, 255), thickness=-1)

    return sheet_music_no_horizontal_lines_no_notes


def remove_lines(sheet_music, horiz_ksize, vert_ksize, show_morph):
    sheet_music_no_horizontal_lines_new = copy.deepcopy(sheet_music)
    # perform opening to remove any horizontal lines
    repair_kernel = cv.getStructuringElement(cv.MORPH_RECT, (horiz_ksize, vert_ksize))
    sheet_music_no_horizontal_lines_new = 255 - cv.morphologyEx(255 - sheet_music_no_horizontal_lines_new,
                                                                cv.MORPH_OPEN, repair_kernel, iterations=1)

    if show_morph:
        cv.imshow("sheet music with no lines", sheet_music_no_horizontal_lines_new)

    return sheet_music_no_horizontal_lines_new


def remove_intro_data(sheet_music, staves_populated, clef_width, accidental_width, time_sig_width):
    sheet_music_no_intro_data = copy.deepcopy(sheet_music)

    for stave in staves_populated:
        x1 = stave.lines[0][0]
        y1 = stave.lines[0][1]
        x2 = stave.lines[len(stave.lines) - 1][0] + clef_width + accidental_width + time_sig_width
        y2 = stave.lines[len(stave.lines) - 1][1]

        cv.rectangle(sheet_music_no_intro_data, [int(x1), int(y1)], [int(x2), int(y2)], color=(255, 255, 255),
                     thickness=-1)

    return sheet_music_no_intro_data


def remove_beams(sheet_music, staves_populated):
    sheet_music_no_beams = copy.deepcopy(sheet_music)

    for stave in staves_populated:
        for beam in stave.beams:
            for note in stave.notes:
                if beam.pt1[0] < note.pt[0] < beam.pt2[0]:
                    cv.rectangle(sheet_music_no_beams, beam.pt1, beam.pt2, color=(255, 255, 255), thickness=-1)

    return sheet_music_no_beams


def fill_in_half_notes(sheet_music, has_half_notes, img_details_size, show_morph):
    sheet_music_with_half_notes_filled = copy.deepcopy(sheet_music)

    if has_half_notes:
        if img_details_size == "small":
            # repair image after horizontal line removal in vertical direction
            repair_kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 1))
            sheet_music_with_half_notes_filled = 255 - cv.morphologyEx(255 - sheet_music_with_half_notes_filled,
                                                                       cv.MORPH_CLOSE, repair_kernel, iterations=2)

            # perform another opening to remove any noise created by dotted notes
            repair_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 3))
            sheet_music_with_half_notes_filled = 255 - cv.morphologyEx(255 - sheet_music_with_half_notes_filled,
                                                                       cv.MORPH_OPEN, repair_kernel, iterations=1)
        elif img_details_size == "medium":
            # repair image after horizontal line removal in vertical direction
            repair_kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 3))
            sheet_music_with_half_notes_filled = 255 - cv.morphologyEx(255 - sheet_music_with_half_notes_filled,
                                                                       cv.MORPH_CLOSE, repair_kernel, iterations=2)

            # perform another opening to remove any noise created by dotted notes
            repair_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 4))
            sheet_music_with_half_notes_filled = 255 - cv.morphologyEx(255 - sheet_music_with_half_notes_filled,
                                                                       cv.MORPH_OPEN, repair_kernel, iterations=1)
        elif img_details_size == "large":
            # repair image after horizontal line removal in vertical direction
            repair_kernel = cv.getStructuringElement(cv.MORPH_RECT, (6, 5))
            sheet_music_with_half_notes_filled = 255 - cv.morphologyEx(255 - sheet_music_with_half_notes_filled,
                                                                       cv.MORPH_CLOSE, repair_kernel, iterations=3)

            # perform another opening to remove any noise created by dotted notes
            repair_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 7))
            sheet_music_with_half_notes_filled = 255 - cv.morphologyEx(255 - sheet_music_with_half_notes_filled,
                                                                       cv.MORPH_OPEN, repair_kernel, iterations=1)

    if show_morph:
        sheet_music_resized = cv.resize(sheet_music_with_half_notes_filled,
                                                       (sheet_music.shape[1] // 2, sheet_music.shape[0] // 2))
        cv.imshow("sheet music with half notes filled", sheet_music_resized)
        cv.waitKey()

    return sheet_music_with_half_notes_filled
