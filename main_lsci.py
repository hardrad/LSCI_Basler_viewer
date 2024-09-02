
from PyQt5.QtWidgets import QApplication
import sys
from lsci import MainWindow


if __name__ == "__main__":
    app = QApplication(sys.argv)
    camera = MainWindow()
    camera.show()
    sys.exit(app.exec())