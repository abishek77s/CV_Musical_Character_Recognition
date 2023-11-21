# CV_Musical_Character_Recognition

# Music Sheet Recognition and Playback System

## Overview
This project presents an innovative solution to music sheet recognition by employing a sophisticated computer vision algorithm. The algorithm combines image recognition and image manipulation techniques to preprocess and generate high-quality images. These enhanced images are then used to train a neural network designed to accurately identify various musical notations such as notes and their type, length, and accents as well as clefts, accidentals, and various rest types.

## Features
- **Advanced Computer Vision Algorithm**: Utilizes a blend of image recognition and manipulation to produce exceptionally clean images.
- **High Accuracy**: Achieves high precision in recognizing note types, lengths, accents, clefs, accidentals, and various rest types.
- **MIDI Note Player Integration**: Converts recognized music sheets into MIDI format, allowing users to audibly playback their chosen music sheets.

## How It Works
1. **Image Preprocessing**: The system first preprocesses the input music sheets to enhance image quality and clarity, preparing them for analysis.
2. **Neural Network Training**: The preprocessed images are fed into the neural network, which is trained to identify and interpret the different elements of music notation.
3. **Music Sheet Analysis**: Once trained, the neural network analyzes new music sheets, identifying the notations with high accuracy.
4. **MIDI Conversion**: The identified notations are then translated into MIDI format, taking into account the nuances of note length, type, and other musical characteristics.
5. **Playback**: Users can play back the converted music in MIDI form, experiencing the music sheet in an auditory format.

## Getting Started
To use this system, please follow the instructions below:

### Prerequisites
- Ensure you have [Python 3.8](https://www.python.org/downloads/) or higher installed.
- Install necessary libraries using `pip install -r requirements.txt`.

### Usage
1. Place your music sheet images in the `input_sheets` directory.
2. Run the main script with `python music_sheet_recognizer.py`.
3. The analyzed MIDI files will be saved in the `output_midi` directory.
4. Use any standard MIDI player to listen to the output.

## Contributions
Contributions are welcome. Please read `CONTRIBUTING.md` for details on our code of conduct, and the process for submitting pull requests.

## License
This project is licensed under the MIT License - see the `LICENSE.md` file for details.
