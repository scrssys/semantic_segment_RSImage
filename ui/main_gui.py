

from PyQt5.QtWidgets import QApplication, QMainWindow
from MainWin import Ui_MainWindow
from PyQt5.QtGui import QIcon
from preProcess.preprocess_implements import child_image_stretch, child_label, child_ImageClip
from sampleProduce.sampleProcess_implements import child_sampleGenCommon
from trainUi.trainModels_implements import child_trainBinaryJaccardCross



class mywindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(mywindow,self).__init__()
        self.move(600,300)
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

    def slot_actiong_image_clip(self):
        child = child_ImageClip()
        child.show()
        child.exec_()

    def slot_action_sampleGenCommon(self):
        child = child_sampleGenCommon()
        child.show()
        child.exec_()

    def slot_action_trainBinaryJaccCross(self):
        child = child_trainBinaryJaccardCross()
        child.show()
        child.exec_()




if __name__=='__main__':
    import sys
    app=QApplication(sys.argv)
    widget=mywindow()
    # widget = child_label()
    widget.show()
    sys.exit(app.exec_())
