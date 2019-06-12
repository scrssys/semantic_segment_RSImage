# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'TrainBinaryJaccardCrossentropy.ui'
#
# Created by: PyQt5 UI code generator 5.5.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog_train_binary_jaccCross(object):
    def setupUi(self, Dialog_train_binary_jaccCross):
        Dialog_train_binary_jaccCross.setObjectName("Dialog_train_binary_jaccCross")
        Dialog_train_binary_jaccCross.resize(400, 300)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog_train_binary_jaccCross)
        self.buttonBox.setGeometry(QtCore.QRect(30, 240, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")

        self.retranslateUi(Dialog_train_binary_jaccCross)
        self.buttonBox.accepted.connect(Dialog_train_binary_jaccCross.accept)
        self.buttonBox.rejected.connect(Dialog_train_binary_jaccCross.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog_train_binary_jaccCross)

    def retranslateUi(self, Dialog_train_binary_jaccCross):
        _translate = QtCore.QCoreApplication.translate
        Dialog_train_binary_jaccCross.setWindowTitle(_translate("Dialog_train_binary_jaccCross", "Dialog"))

