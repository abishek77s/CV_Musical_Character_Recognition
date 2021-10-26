class Note:
    def __init__(self, name, octave, length, accidental, pt, size):
        self.name = name
        self.octave = octave
        self.length = length
        self.accidental = accidental
        self.pt = pt
        self.size = size

    def to_str(self):
        print("Note Information")
        print("Name:", self.name)
        print("Octave:", self.octave)
        print("Length:", self.length)
        print("Accidental:", self.accidental)
        print("Location:", self.pt)
        print("Size:", self.size)
