class Accidental:
    def __init__(self, note, type, pt, size):
        self.note = note
        self.type = type
        self.pt = pt
        self.size = size

    def to_str(self):
        print("Accidental Information")
        print("Note:", self.note)
        print("Type:", self.type)
        print("Location:", self.pt)
        print("Size:", self.size)
