class TmpWriter:
    def __init__(self, path: str):
        self.path = path
        self.lines = []

    def writeln(self, content: str):
        self.lines.append(content + "\n")

        if len(self.lines) >= 10:
            self.save()

    def save(self):
        with open(self.path, "a", encoding="ascii") as file:
            file.writelines(self.lines)

        self.lines.clear()