# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainWin.ui'
#
# Created by: PyQt5 UI code generator 5.5.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(515, 406)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 515, 19))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuPrepocess = QtWidgets.QMenu(self.menubar)
        self.menuPrepocess.setObjectName("menuPrepocess")
        self.menuTrain = QtWidgets.QMenu(self.menubar)
        self.menuTrain.setObjectName("menuTrain")
        self.menuClassify = QtWidgets.QMenu(self.menubar)
        self.menuClassify.setObjectName("menuClassify")
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionLabel_check = QtWidgets.QAction(MainWindow)
        self.actionLabel_check.setObjectName("actionLabel_check")
        self.menuPrepocess.addAction(self.actionLabel_check)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuPrepocess.menuAction())
        self.menubar.addAction(self.menuTrain.menuAction())
        self.menubar.addAction(self.menuClassify.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(MainWindow)
        self.actionLabel_check.triggered.connect(MainWindow.for_action_label_check)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuPrepocess.setTitle(_translate("MainWindow", "Prepocess"))
        self.menuTrain.setTitle(_translate("MainWindow", "Train"))
        self.menuClassify.setTitle(_translate("MainWindow", "Classify"))
        self.menuHelp.setTitle(_translate("MainWindow", "Help"))
        self.actionLabel_check.setText(_translate("MainWindow", "Label_check"))

