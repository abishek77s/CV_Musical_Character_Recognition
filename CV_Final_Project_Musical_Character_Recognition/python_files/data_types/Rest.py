class Rest:
    def __init__(self, pt, type):
        self.pt = pt
        self.type = type

    def to_str(self):
        print("Rest Information")
        print("Point:", self.pt)
        print("Type:", self.type)