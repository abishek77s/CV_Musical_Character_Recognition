class Stave:
    def __init__(self, lines, line_spacing, clef, accidentals, time_signature, notes, beams, rests):
        self.lines = lines
        self.line_spacing = line_spacing
        self.clef = clef
        self.accidentals = accidentals
        self.time_signature = time_signature
        self.notes = notes
        self.beams = beams
        self.rests = rests

    def to_str(self):
        print("Stave Information")
        print("Clef:", self.clef)
        print("Accidentals", self.accidentals)
        print("Time Signature:", self.time_signature)
        print("Notes:", self.notes)
        print("Beams:", self.beams)
        print("Rests:", self.rests)
