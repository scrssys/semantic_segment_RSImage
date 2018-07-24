# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'label_check.ui'
#
# Created by: PyQt5 UI code generator 5.5.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog_label_check(object):
    def setupUi(self, Dialog_label_check):
        Dialog_label_check.setObjectName("Dialog_label_check")
        Dialog_label_check.resize(400, 300)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog_label_check)
        self.buttonBox.setGeometry(QtCore.QRect(30, 240, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.lineEdit = QtWidgets.QLineEdit(Dialog_label_check)
        self.lineEdit.setGeometry(QtCore.QRect(100, 30, 171, 22))
        self.lineEdit.setObjectName("lineEdit")
        self.label = QtWidgets.QLabel(Dialog_label_check)
        self.label.setGeometry(QtCore.QRect(20, 30, 71, 16))
        self.label.setObjectName("label")
        self.pushButton = QtWidgets.QPushButton(Dialog_label_check)
        self.pushButton.setGeometry(QtCore.QRect(290, 30, 80, 22))
        self.pushButton.setObjectName("pushButton")

        self.retranslateUi(Dialog_label_check)
        self.buttonBox.accepted.connect(Dialog_label_check.accept)
        self.buttonBox.rejected.connect(Dialog_label_check.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog_label_check)

    def retranslateUi(self, Dialog_label_check):
        _translate = QtCore.QCoreApplication.translate
        Dialog_label_check.setWindowTitle(_translate("Dialog_label_check", "Dialog"))
        self.label.setText(_translate("Dialog_label_check", "label image"))
        self.pushButton.setText(_translate("Dialog_label_check", "Open"))

