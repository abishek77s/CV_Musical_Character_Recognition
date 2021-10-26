class Beam:
    def __init__(self, pt1, pt2):
        self.pt1 = pt1
        self.pt2 = pt2

    def to_str(self):
        print("Beam Information")
        print("Point 1:", self.pt1)
        print("Point 2:", self.pt2)