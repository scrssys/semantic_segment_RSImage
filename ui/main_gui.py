

from PyQt5.QtWidgets import QApplication, QDialog, QMainWindow
from MainWin import Ui_MainWindow
from label_check import Ui_Dialog_label_check
from PyQt5.QtGui import QIcon
from ImageStretch_implements import child_image_stretch


class mywindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(mywindow,self).__init__()
        self.setWindowTitle('Image Interpretation based-on Deep Learning')
        self.setWindowIcon(QIcon('scrslogo.png'))
        self.setupUi(self)

    def for_action_label_check(self):
        child = child_label()
        child.show()
        child.exec_()

    def for_action_image_stretch(self):
        child = child_image_stretch()
        child.show()
        child.exec_()


class child_label(QDialog, Ui_Dialog_label_check):
    def __init__(self):
        super(child_label, self).__init__()
        self.setupUi(self)

# class child_image_stretch(QDialog, Ui_Dialog_image_stretch):
#     def __init__(self):
#         super(child_image_stretch,self).__init__()
#         self.setupUi(self)
#
#     def slot_select_input_dir(self):
#         dir_tmp = QFileDialog.getExistingDirectory(self, "select a existing directory", '../../data/')
#         self.lineEdit_input.setText(dir_tmp)
#
#     def slot_select_output_dir(self):
#         dir_tmp = QFileDialog.getExistingDirectory(self, "select a existing directory", '../../data/')
#         self.lineEdit_output.setText(dir_tmp)
#
#     def slot_ok(self):
#         input_dir = self.lineEdit_input.text()
#         output_dir = self.lineEdit_output.text()




if __name__=='__main__':
    import sys
    app=QApplication(sys.argv)
    widget=mywindow()
    # widget = child_label()
    widget.show()
    sys.exit(app.exec_())
