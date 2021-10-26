import copy
import random
import numpy as np

import cv2 as cv


def draw_horizontal_lines(sheet_music, staves_populated, resize_img, dim):
    sheet_music_with_lines_highlighted = copy.deepcopy(sheet_music)
    # display remaining horizontal lines
    for stave in staves_populated:
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        for line in stave.lines:
            x1, y1, x2, y2 = line
            cv.line(sheet_music_with_lines_highlighted, (x1, y1), (x2, y2), (r, g, b), 2)

    if resize_img:
        sheet_music_with_lines_highlighted = cv.resize(sheet_music_with_lines_highlighted, dim)
    return sheet_music_with_lines_highlighted


def draw_clef(sheet_music, staves_populated, clef_width, resize_img, dim):
    sheet_music_with_clef = copy.deepcopy(sheet_music)

    for stave in staves_populated:
        x1 = stave.lines[0][0]
        y1 = stave.lines[0][1]
        x2 = stave.lines[len(stave.lines) - 1][0] + clef_width
        y2 = stave.lines[len(stave.lines) - 1][1]

        # draw rectangle around clef for visualization
        cv.rectangle(sheet_music_with_clef, (x1, y1), (x2, y2), (255, 0, 0))
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(sheet_music_with_clef, stave.clef, (x2, y2), font, 0.8, (255, 0, 0))

    if resize_img:
        sheet_music_with_clef = cv.resize(sheet_music_with_clef, dim)
    return sheet_music_with_clef


def draw_accidentals(sheet_music, staves_populated, resize_img, dim):
    sheet_music_with_accidentals = copy.deepcopy(sheet_music)

    for stave in staves_populated:
        for accidental in stave.accidentals:
            font = cv.FONT_HERSHEY_SIMPLEX
            cv.putText(sheet_music_with_accidentals, accidental.note + " " + accidental.type,
                       (int(accidental.pt[0]), int(accidental.pt[1])), font, 0.8, (255, 0, 0))

    if resize_img:
        sheet_music_with_accidentals = cv.resize(sheet_music_with_accidentals, dim)
    return sheet_music_with_accidentals


def draw_time_signature(sheet_music, staves_populated, clef_width, accidental_width, time_sig_width, resize_img, dim):
    sheet_music_with_time_signature = copy.deepcopy(sheet_music)

    for stave in staves_populated:
        if stave.time_signature is not None:
            x1 = stave.lines[0][0] + clef_width + accidental_width
            y1 = stave.lines[0][1]
            x2 = stave.lines[0][0] + clef_width + accidental_width + time_sig_width
            y2 = stave.lines[len(stave.lines) - 1][1]

            font = cv.FONT_HERSHEY_SIMPLEX
            cv.rectangle(sheet_music_with_time_signature, (x1, y1), (x2, y2), (255, 0, 0))
            cv.putText(sheet_music_with_time_signature, stave.time_signature,
                       (int(x2), int(y2)), font, 0.8, (255, 0, 0))

    if resize_img:
        sheet_music_with_time_signature = cv.resize(sheet_music_with_time_signature, dim)
    return sheet_music_with_time_signature


def draw_blobs(sheet_music, note_blobs, resize_img, dim):
    sheet_music_with_note_blobs = copy.deepcopy(sheet_music)

    # Draw blobs
    sheet_music_with_note_blobs = cv.drawKeypoints(sheet_music_with_note_blobs, note_blobs, np.array([]), (0, 0, 255),
                                       cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    if resize_img:
        sheet_music_with_note_blobs = cv.resize(sheet_music_with_note_blobs, dim)
    return sheet_music_with_note_blobs


def draw_notes(sheet_music, staves_populated, resize_img, dim):
    sheet_music_with_notes = copy.deepcopy(sheet_music)

    for stave in staves_populated:
        for note in stave.notes:
            font = cv.FONT_HERSHEY_SIMPLEX
            print(note.octave)
            cv.putText(sheet_music_with_notes, note.name + str(note.octave),
                       (int(note.pt[0]), int(note.pt[1])), font, 0.8, (255, 0, 0))

    if resize_img:
        sheet_music_with_notes = cv.resize(sheet_music_with_notes, dim)
    return sheet_music_with_notes


def draw_note_lengths(sheet_music, staves_populated, resize_img, dim):
    sheet_music_with_note_lengths = copy.deepcopy(sheet_music)

    for stave in staves_populated:
        for note in stave.notes:
            font = cv.FONT_HERSHEY_SIMPLEX
            cv.putText(sheet_music_with_note_lengths, note.length[0],
                       (int(note.pt[0]), int(note.pt[1])), font, 0.8, (255, 0, 0))

    if resize_img:
        sheet_music_with_note_lengths = cv.resize(sheet_music_with_note_lengths, dim)
    return sheet_music_with_note_lengths


def draw_rests(sheet_music, staves_populated, resize_img, dim):
    sheet_music_with_rests = copy.deepcopy(sheet_music)

    for stave in staves_populated:
        for rest in stave.rests:
            font = cv.FONT_HERSHEY_SIMPLEX
            cv.putText(sheet_music_with_rests, rest.type[0],
                       (int(rest.pt[0]), int(rest.pt[1])), font, 0.8, (255, 0, 0))

    if resize_img:
        sheet_music_with_rests = cv.resize(sheet_music_with_rests, dim)
    return sheet_music_with_rests
