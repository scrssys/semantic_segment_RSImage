

from PyQt5.QtWidgets import QApplication, QWidget, QDialog, QMainWindow
from MainWin import Ui_MainWindow
from label_check import Ui_Dialog_label_check

class mywindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(mywindow,self).__init__()
        self.setupUi(self)
    def for_action_label_check(self):
        child = child_label()
        child.show()
        child.exec_()


class child_label(QDialog, Ui_Dialog_label_check):
    def __init__(self):
        super(child_label, self).__init__()
        self.setupUi(self)



if __name__=='__main__':
    import sys
    app=QApplication(sys.argv)
    widget=mywindow()
    widget.show()
    sys.exit(app.exec_())
