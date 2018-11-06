# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'about.ui'
#
# Created by: PyQt5 UI code generator 5.5.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog_about(object):
    def setupUi(self, Dialog_about):
        Dialog_about.setObjectName("Dialog_about")
        Dialog_about.resize(353, 143)
        self.textEdit = QtWidgets.QTextEdit(Dialog_about)
        self.textEdit.setGeometry(QtCore.QRect(-10, 0, 371, 151))
        self.textEdit.setObjectName("textEdit")

        self.retranslateUi(Dialog_about)
        QtCore.QMetaObject.connectSlotsByName(Dialog_about)

    def retranslateUi(self, Dialog_about):
        _translate = QtCore.QCoreApplication.translate
        Dialog_about.setWindowTitle(_translate("Dialog_about", "About"))
        self.textEdit.setHtml(_translate("Dialog_about", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Sans Serif\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">    <img src=\":/log/scrslogo.png\" />   <span style=\" font-size:16pt;\">Copyright SCRS</span><span style=\" font-size:20pt;\"> </span></p></body></html>"))

import mysrc_rc
