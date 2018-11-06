# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'VoteMultimodleResults.ui'
#
# Created by: PyQt5 UI code generator 5.5.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog_vote_multimodels(object):
    def setupUi(self, Dialog_vote_multimodels):
        Dialog_vote_multimodels.setObjectName("Dialog_vote_multimodels")
        Dialog_vote_multimodels.resize(489, 300)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog_vote_multimodels)
        self.buttonBox.setGeometry(QtCore.QRect(30, 240, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.layoutWidget = QtWidgets.QWidget(Dialog_vote_multimodels)
        self.layoutWidget.setGeometry(QtCore.QRect(20, 150, 401, 25))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_7 = QtWidgets.QLabel(self.layoutWidget)
        self.label_7.setMinimumSize(QtCore.QSize(55, 23))
        self.label_7.setObjectName("label_7")
        self.horizontalLayout.addWidget(self.label_7)
        self.lineEdit_mask = QtWidgets.QLineEdit(self.layoutWidget)
        self.lineEdit_mask.setMinimumSize(QtCore.QSize(201, 23))
        self.lineEdit_mask.setObjectName("lineEdit_mask")
        self.horizontalLayout.addWidget(self.lineEdit_mask)
        self.pushButton_mask = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton_mask.setMinimumSize(QtCore.QSize(0, 23))
        self.pushButton_mask.setObjectName("pushButton_mask")
        self.horizontalLayout.addWidget(self.pushButton_mask)
        self.layoutWidget_3 = QtWidgets.QWidget(Dialog_vote_multimodels)
        self.layoutWidget_3.setGeometry(QtCore.QRect(10, 20, 411, 27))
        self.layoutWidget_3.setObjectName("layoutWidget_3")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout(self.layoutWidget_3)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.label_6 = QtWidgets.QLabel(self.layoutWidget_3)
        self.label_6.setMinimumSize(QtCore.QSize(55, 23))
        self.label_6.setObjectName("label_6")
        self.horizontalLayout_8.addWidget(self.label_6)
        self.lineEdit_inputs = QtWidgets.QLineEdit(self.layoutWidget_3)
        self.lineEdit_inputs.setMinimumSize(QtCore.QSize(201, 23))
        self.lineEdit_inputs.setObjectName("lineEdit_inputs")
        self.horizontalLayout_8.addWidget(self.lineEdit_inputs)
        self.pushButton_inputs = QtWidgets.QPushButton(self.layoutWidget_3)
        self.pushButton_inputs.setMinimumSize(QtCore.QSize(0, 23))
        self.pushButton_inputs.setObjectName("pushButton_inputs")
        self.horizontalLayout_8.addWidget(self.pushButton_inputs)
        self.groupBox = QtWidgets.QGroupBox(Dialog_vote_multimodels)
        self.groupBox.setGeometry(QtCore.QRect(50, 60, 311, 80))
        self.groupBox.setObjectName("groupBox")
        self.spinBox_max = QtWidgets.QSpinBox(self.groupBox)
        self.spinBox_max.setGeometry(QtCore.QRect(200, 40, 47, 23))
        self.spinBox_max.setObjectName("spinBox_max")
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setGeometry(QtCore.QRect(10, 40, 41, 16))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setGeometry(QtCore.QRect(150, 40, 41, 16))
        self.label_2.setObjectName("label_2")
        self.spinBox_min = QtWidgets.QSpinBox(self.groupBox)
        self.spinBox_min.setGeometry(QtCore.QRect(60, 40, 47, 23))
        self.spinBox_min.setObjectName("spinBox_min")

        self.retranslateUi(Dialog_vote_multimodels)
        self.buttonBox.accepted.connect(Dialog_vote_multimodels.accept)
        self.pushButton_inputs.clicked.connect(Dialog_vote_multimodels.slot_select_input_files)
        self.pushButton_mask.clicked.connect(Dialog_vote_multimodels.slot_get_save_mask)
        self.buttonBox.accepted.connect(Dialog_vote_multimodels.slot_ok)
        QtCore.QMetaObject.connectSlotsByName(Dialog_vote_multimodels)

    def retranslateUi(self, Dialog_vote_multimodels):
        _translate = QtCore.QCoreApplication.translate
        Dialog_vote_multimodels.setWindowTitle(_translate("Dialog_vote_multimodels", "Dialog"))
        self.label_7.setText(_translate("Dialog_vote_multimodels", "Mask:"))
        self.pushButton_mask.setText(_translate("Dialog_vote_multimodels", "Open"))
        self.label_6.setText(_translate("Dialog_vote_multimodels", "Images:"))
        self.pushButton_inputs.setText(_translate("Dialog_vote_multimodels", "Open"))
        self.groupBox.setTitle(_translate("Dialog_vote_multimodels", "Values range"))
        self.label.setText(_translate("Dialog_vote_multimodels", "min:"))
        self.label_2.setText(_translate("Dialog_vote_multimodels", "max:"))

