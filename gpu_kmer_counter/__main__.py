# Code for user interface 
# Author: Noah Legall
# Team: John Lees, Louise Cerdeira, Sam Horsfield, Noah Legall

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QWidget, QPushButton, QApplication, QGridLayout, QLabel, QFrame, QFileDialog
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys

# initializes the window
def window():
    # compulsory lines
    app = QApplication(sys.argv)
    win = QMainWindow()

    # set window dimensions and window title
    # x coord, y coord, width, height
    win.setGeometry(400,400,500,300)
    win.setWindowTitle("kmer-counter")
 
    # actually show the window
    win.show()

    # wait for the user to click 'x' to close the window
    sys.exit(app.exec_())
 
class PyQtLayout(QWidget):
 
    def __init__(self):
        super().__init__()
        self.UI()
        self.setFixedHeight(300)
        self.setFixedWidth(700)
 
    def UI(self):
        # Text labels for dialog boxes.
        label1 = QLabel("input directory:")
        label2 = QLabel("k-mer length (bps):")
        labelGuess = QLabel("guess # of k-mers")
        label3 = QLabel("memory available (MB):")
        label4 = QLabel("false positivity rate (0-1):")

        self.line1 = QtWidgets.QLineEdit()
        self.line1.setFixedWidth(400)
        self.line2 = QtWidgets.QLineEdit()
        self.line2.setFixedWidth(150)
        self.line3 = QtWidgets.QLineEdit()
        self.line3.setFixedWidth(150)
        self.line4 = QtWidgets.QLineEdit()
        self.line4.setFixedWidth(150)  
        self.lineGuess = QtWidgets.QLineEdit()
        self.lineGuess.setFixedWidth(150)

        launch = QPushButton() 
        launch.setText("Submit")
        launch.clicked.connect(self.extract)
        browseLaunch = QPushButton() 
        browseLaunch.setText("...")
        browseLaunch.clicked.connect(self.browse)     
         
        grid = QGridLayout()
        grid.addWidget(label1, 0, 0)
        grid.addWidget(label2, 1, 0)
        grid.addWidget(label3, 2, 0)
        grid.addWidget(label4, 3, 0)
        grid.addWidget(labelGuess,4,0)
        grid.addWidget(self.line1, 0, 1)
        grid.addWidget(self.line2, 1, 1)
        grid.addWidget(self.line3, 2, 1)
        grid.addWidget(self.line4, 3, 1)
        grid.addWidget(self.lineGuess,4,1)
        grid.addWidget(launch, 5, 0)
        grid.addWidget(browseLaunch,0,3)
        
        self.setLayout(grid)
        self.setGeometry(300, 300, 500, 200)
        self.setWindowTitle('kmer-counter')
        self.show()
    
    def browse(self):
        response = QFileDialog.getExistingDirectory(
            self, 
            caption='Select Folder'
        )
        print(response)
        self.line1.setText(response)
        return response
    
    def extract(self):
        seq_dir = self.line1.text()
        k = int(self.line2.text())
        k_guess = int(self.lineGuess.text())
        mem = int(self.line3.text())
        fpr = float(self.line4.text())

        if not isinstance(k, int) or not isinstance(fpr,float):
            sys.exit(0)
        else:
            print(
"""
Testing to see if values can be extracted: 
{}
{}
{}
{}
{}
""".format(seq_dir,k,k_guess,mem,fpr))


def main():
    app = QApplication(sys.argv)
    ex = PyQtLayout()
    sys.exit(app.exec_())
 
if __name__ == '__main__':
    main()