from ConvNet import *
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QMessageBox

class Ui_MainWindow(object):
    def __init__(self):
        self.conv = MainConv()
        
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1161, 917)
        MainWindow.setStyleSheet("background-color: rgb(32, 48, 64)")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.dataSetButton = QtWidgets.QPushButton(self.centralwidget)
        self.dataSetButton.setGeometry(QtCore.QRect(40, 520, 311, 271))
        self.dataSetButton.setAutoFillBackground(False)
        self.dataSetButton.setStyleSheet("background-color : rgb(45, 66, 87);\n"
"color: white;\n"
"font-size: 50;")
        self.dataSetButton.setDefault(False)
        self.dataSetButton.setFlat(False)
        self.dataSetButton.setObjectName("dataSetButton")
        self.testButton = QtWidgets.QPushButton(self.centralwidget)
        self.testButton.setGeometry(QtCore.QRect(410, 520, 311, 271))
        self.testButton.setAutoFillBackground(False)
        self.testButton.setStyleSheet("background-color : rgb(45, 66, 87);\n"
"color : white;")
        self.testButton.setObjectName("testButton")
        
        self.trainNeuralNetwork = QtWidgets.QPushButton(self.centralwidget)
        self.trainNeuralNetwork.setGeometry(QtCore.QRect(770, 520, 311, 271))
        self.trainNeuralNetwork.setStyleSheet("background-color : rgb(45, 66, 87);\n"
"color : white;")
        self.trainNeuralNetwork.setObjectName("trainNeuralNetwork")
        self.trainNeuralNetwork.clicked.connect(self.clickTestModel)
        self.verticalWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalWidget.setGeometry(QtCore.QRect(0, -10, 1131, 101))
        self.verticalWidget.setStyleSheet("background-color: rgb(255, 255, 255)")
        self.verticalWidget.setObjectName("verticalWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(self.verticalWidget)
        font = QtGui.QFont()
        font.setFamily("Gulim")
        font.setPointSize(40)
        self.label.setFont(font)
        self.label.setStyleSheet("background-color : rgb(31, 47, 62)\n"
"")
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.verticalWidget1 = QtWidgets.QWidget(self.centralwidget)
        self.verticalWidget1.setGeometry(QtCore.QRect(0, 400, 1121, 491))
        self.verticalWidget1.setObjectName("verticalWidget1")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalWidget1)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalWidget.raise_()
        self.verticalWidget.raise_()
        self.dataSetButton.raise_()
        self.testButton.raise_()
        self.trainNeuralNetwork.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1161, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.testButton.clicked.connect(self.buildModel)
        self.dataSetButton.clicked.connect(self.clickImportMat)
        
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.dataSetButton.setText(_translate("MainWindow", "Import Dataset"))
        self.trainNeuralNetwork.setText(_translate("MainWindow", "Predict"))
        self.testButton.setText(_translate("MainWindow", "Train neural network"))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" color:#ffffff;\">Diagnosing Alzheimer\'s Disease </span></p></body></html>"))

    def buildModel(self):
        if self.conv.getTrained() == True:
            QMessageBox.about(None, "ERROR #4", "Model has already been trained!")
        else:
            if(len(self.conv.getTrain_Images()) != 0):
                self.conv.initaliseNetwork()
                self.conv.train(1)
            else:
                QMessageBox.about(None, "ERROR #1", "Training set and training labels have not been selected!")

                   
    
    def clickTestModel(self):
        if(self.conv.getTrained() == True):
            fname = ""
            while fname == "":
                fname = QFileDialog.getOpenFileName(None, 'Open file', 'c:\\',"Image files (*.jpg *.gif)")
            output = self.conv.predict(fname)
            if np.argmax(output) == 0:
                QMessageBox.about(None, "RESULTS", "Alzhimer's Disease is not present")

            elif np.argmax(output) == 1:
                QMessageBox.about(None, "RESULTS", "Early signs of Alzhimer's Disease are present")
                
            else:
                QMessageBox.about(None, "RESULTS", "Alzhimer's Disease is present")

        else:
            QMessageBox.about(None, "ERROR #2", "Model has not been trained!")


    def clickImportMat(self):
        ok = False
        inputMat = ('', '')
        labelMat = ('', '')
        if len(self.conv.getTrain_Images()) > 0:
            QMessageBox.about(None, "ERROR #5", "You have already selected a data set and labels")
        else:
            while(ok == False):
                while inputMat == ('', ''):
                    QMessageBox.about(None, "Select a file", "SELECT THE DATASET")
                    inputMat = QFileDialog.getOpenFileName(None, 'Open file', 'c:\\',"Mat files (*.mat)")
                while labelMat == ('', ''):
                    QMessageBox.about(None, "Select a file", "SELECT THE LABELS")
                    labelMat = QFileDialog.getOpenFileName(None, 'Open file', 'c:\\',"Mat files (*.mat)")
                x = self.conv.importCustomMat(inputMat)
                y = self.conv.importCustomMatLabel(labelMat)
                if(len(x) != len(y)):
                    QMessageBox.about(None, "ERROR #3", "Data Set and set of labels are not the same size! Select another set...")
                else:
                    ok = True
        #self.conv.test()
        
        
        

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
