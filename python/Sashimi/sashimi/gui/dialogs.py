from PySide6.QtWidgets import QFileDialog

class CNNDirectoryDialog(QFileDialog):
    def __init__(self):
        super().__init__()
        self.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
        self.setFileMode(QFileDialog.FileMode.Directory)

