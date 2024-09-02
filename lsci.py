

from PyQt5.QtWidgets import *
#from PyQt5.QtGui import *
from PyQt5.QtCore import *
from ui_lsci import Ui_MainWindow
#from MlpWidgetClass import MatplotlibWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
#import sys
from pypylon import pylon
import cv2
import time
import matplotlib
import tifffile
#import numpy as np
import os
from datetime import datetime, date
import time
matplotlib.use('Qt5Agg')
# fsdfsdf
import numpy as np
from scipy.ndimage import uniform_filter
import pyqtgraph as pg
import dask.array as da

pg.setConfigOptions(imageAxisOrder='row-major')



class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)
        fig.tight_layout()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()



        self._ui = Ui_MainWindow()
        self._ui.setupUi(self)
        self.current_date = date.today()
        self._ui.lineEdit_4.setText(self.current_date.strftime("%d/%m/%Y"))

        self.project_path = os.getcwd()
        self._ui.lineEdit_5.setText(self.project_path)
        self._ui.verticalSlider.setTickPosition(QSlider.TickPosition.TicksRight)
        self.qslider_range = np.arange(0,1.01,0.01)

        # Emulation initialising
        if len(pylon.TlFactory.GetInstance().EnumerateDevices()) == 0:
            self.camera_emu = 1
            os.environ["PYLON_CAMEMU"] = "1"
            self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
            self.camera.Open()
            img_dir = 'C:\\Users\\IK\\pycharmProjects\\ui_project\\lsci_gui\\img_dir'
            self.camera.ImageFilename = img_dir

            # enable image file test pattern
            self.camera.ImageFileMode = "On"
            # disable testpattern [ image file is "real-image"]
            self.camera.TestImageSelector = "Off"
            # choose one pixel format. camera emulation does conversion on the fly
            self.camera.PixelFormat = "Mono8"

            # set camera width and height
            self.camera.Width.SetValue(2048)
            self.camera.Height.SetValue(2048)
            #camera.ExposureMode.SetValue(Timed);
            self.camera.ExposureTimeAbs.SetValue(300.0)  # 0.3 ms
            self.camera.AcquisitionFrameRateAbs.SetValue(1.0)  # 10 fps
            self.camera.AcquisitionFrameRateEnable.SetValue = True
            self.framerate = float(self.camera.AcquisitionFrameRateAbs.GetValue())

            self._ui.lineEdit.setText(
                "" + "{:.1f}".format(self.camera.AcquisitionFrameRateAbs.GetValue()))
            self._ui.lineEdit_2.setText(
                "" + "{:.3f}".format(self.camera.ExposureTimeAbs.GetValue()))
            self._ui.lineEdit_3.setText("Camera Type: " + self.camera.GetDeviceInfo().GetModelName())

            self._ui.lineEdit_6.setText(str(0.0))
            self._ui.lineEdit_7.setText(str(1.0))
            self.color_min = float("{:.2f}".format(float(self._ui.lineEdit_6.text())))
            self.color_max = float("{:.2f}".format(float(self._ui.lineEdit_7.text())))
            self._ui.verticalSlider.setHigh(100)


        # Real camera initialize
        else:
            self.camera_emu = 0
            self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
            self.camera.Open()
            self.camera.PixelFormat = "Mono12"
            # Set default values for exposure time and frame rate
            self.camera.AcquisitionFrameRate.SetValue(1.0)  # 1 fps
            self.camera.AcquisitionFrameRateEnable.SetValue = True
            self.camera.ExposureTime.SetValue(42.0)  # 30 mus

            self.framerate = float(self.camera.AcquisitionFrameRate.GetValue())
            self._ui.lineEdit.setText(
                "" + "{:.1f}".format(self.camera.AcquisitionFrameRate.GetValue()))
            self._ui.lineEdit_2.setText(
                "" + "{:.1f}".format(self.camera.ExposureTime.GetValue()))
            self._ui.lineEdit_3.setText("Camera Type: " + self.camera.GetDeviceInfo().GetModelName())

            self._ui.lineEdit_6.setText(str(0.0))
            self._ui.lineEdit_7.setText(str(1.0))

            self.color_min = float("{:.2f}".format(float(self._ui.lineEdit_6.text())))
            self.color_max = float("{:.2f}".format(float(self._ui.lineEdit_7.text())))
            self._ui.verticalSlider.setHigh(100)

        # pyqtgraph widget
        self.im_widget = pg.ImageView(self)
        self.im_widget.ui.histogram.hide()
        self.im_widget.ui.roiBtn.hide()
        self.im_widget.ui.menuBtn.hide()
        self.cm = pg.colormap.get('jet_r', source='matplotlib')

        # pyqtgraph widget


        self.timer = QTimer()
        self.timer.timeout.connect(self.update_image)

        #self.scene = QGraphicsScene(self._ui.graphicsView)
        #self._ui.graphicsView.setScene(self.scene)

        #self.mlp_widget = MatplotlibWidget()
        #self.proxy = QGraphicsProxyWidget()
        #self.proxy.setWidget(self.mlp_widget)

        #self.scene.addItem(self.proxy)
        #self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self._ui.gridLayout_6.addWidget(self.im_widget, 2, 1, 1, 1)


    def slider_changed(self):

        local_min = self._ui.verticalSlider.low()
        local_max = self._ui.verticalSlider.high()

        self._ui.lineEdit_6.setText("{:.2f}".format(self.qslider_range[local_min]))
        self._ui.lineEdit_7.setText("{:.2f}".format(self.qslider_range[local_max]))

        self.color_min = self.qslider_range[local_min]
        self.color_max = self.qslider_range[local_max]


    def start_recording(self):
        # Start the camera and the image timer
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self.timer_interval = int(1000 / self.framerate)   # Update the image every int(1000 / self.framerate)
        self.timer.start(self.timer_interval)
        self._ui.pushButton.setEnabled(False)
        self._ui.pushButton_2.setEnabled(True)
        self.count = 0

    def stop_recording(self):
        # Stop the camera and the image timer
        self.timer.stop()
        self.camera.StopGrabbing()
        self.camera.Close()
        self._ui.pushButton.setEnabled(True)
        self._ui.pushButton_2.setEnabled(False)


    def set_colorMax(self):

        if self.color_max > self.color_min:
            self.color_max = float("{:.2f}".format(float(self._ui.lineEdit_7.text())))
            self._ui.verticalSlider.setHigh(int(self.color_max*100.0))
        else:
            self.color_max = self.color_min + 0.01
            self._ui.lineEdit_7.setText(str(self.color_max))
            self._ui.verticalSlider.setHigh(int(self.color_max * 100.0))


    def set_colorMin(self):

        if self.color_min < self.color_max:
            self.color_min = float("{:.2f}".format(float(self._ui.lineEdit_6.text())))
            self._ui.verticalSlider.setLow(int(self.color_min*100.0))
        else:
            self.color_min = self.color_max - 0.01
            self._ui.lineEdit_6.setText(format(self.color_min,".2f"))
            self._ui.verticalSlider.setLow(int(self.color_min * 100.0))

    def set_framerate(self):
        # Set the camera frame rate to the value entered in the editable window
        self.framerate = float(self._ui.lineEdit.text())
        if self.camera_emu == 1:
            self.camera.AcquisitionFrameRateAbs.SetValue(self.framerate)
        else:
            self.camera.AcquisitionFrameRate.SetValue(self.framerate)
        self.timer_interval = int(1000 /self.framerate)  # Update the timer interval
        self.timer.setInterval(self.timer_interval)

    def set_exposure(self):
        # Set the camera exposure time to the value entered in the editable window
        self.exposure = float(self._ui.lineEdit_2.text())
        if self.camera_emu == 1:
            self.camera.ExposureTimeAbs.SetValue(self.exposure)
        else:
            self.camera.ExposureTime.SetValue(self.exposure)

    def conv_mean(self,img, size=7):
        "ndimage.uniform_filter with `size=7`"

        # uniform_filter create an already averaged result/
        return uniform_filter(img, size=7,output=np.float32)
        # return signal.convolve2d(img, np.ones((size, size)), boundary='symm', mode='same')

    def lsci_processing(self,img, size=7):

        mu_x = self.conv_mean(img)  # / (size ** 2)

        # Factor need here for computing the corrected
        # variance computation with 1/(N-1) factor, according to uniform_filter features
        x_sq = self.conv_mean(np.float32(img) ** 2) * ((size ** 2) / ((size ** 2) - 1))
        var_x = x_sq - np.square(mu_x) * ((size ** 2) / ((size ** 2) - 1))
        #sc = gaussian_filter((np.sqrt(np.abs(var_x)) / mu_x), 3)
        sc = np.sqrt(np.abs(var_x)) / mu_x

        return sc



    def speckle_contrast_proc(self,img):
        #data = img.asarray()
        chunk_size = [x // 2 for x in img.shape]

        img_da = da.from_array(img, chunks=chunk_size)

        # Process the data as needed

        return img_da.map_overlap(self.lsci_processing, depth=50, dtype='float32').compute()


    def save_tiff(self):
        self._ui.checkBox.setStyleSheet("QCheckBox::indicator"
                                        "{"
                                        "background-color : red;"
                                        "}")
        now_date = datetime.now()
        if self._ui.checkBox.isChecked():
            self.current_tiff_file = f"stacked_images_{round(time.mktime(now_date.timetuple()))}.tiff"
            self._ui.checkBox.setStyleSheet("QCheckBox::indicator"
                                        "{"
                                        "background-color : lightgreen;"
                                        "}")

            #stacked_tiff = tifffile.TiffWriter(self.project_path + '\\' + current_tiff_file, bigtiff=True)

    def set_path(self):
        self.project_path = self._ui.lineEdit_5.text()



    def update_image(self):
        # Placeholder method to update the image in the QGraphicsView
        self.count +=1
        start = time.time()
        grab_result = self.camera.RetrieveResult(2000) # Replace with the actual path to the image
        if grab_result.GrabSucceeded():

            image_orig = grab_result.Array #+ np.random.normal(0,8,(grab_result.Array.shape[0],grab_result.Array.shape[1]))
            image = self.speckle_contrast_proc(image_orig)


            image_scaled = cv2.resize(image, (300, 300), interpolation=cv2.INTER_LINEAR)

            #qimage = QImage(image_scaled.data, image_scaled.shape[1], image_scaled.shape[0], QImage.Format_Grayscale8)
            #pixmap = QPixmap.fromImage(qimage)

            #self.scene.addPixmap((pixmap))
            #self._ui.graphicsView.setScene(self.scene)

            #self.canvas.axes.clear()
            #im = self.canvas.axes.imshow(image_scaled, vmin=0, vmax=0.5,
            #                                cmap='jet_r', origin='upper',)
            #                                #interpolation='bilinear')
            #cb = self.canvas.fig.colorbar(im)
            self.im_widget.setImage(image_scaled, levels=[self.color_min, self.color_max])
            #self.im_widget.setLevels(0, 0.5)
            self.im_widget.setColorMap(self.cm)
            self.im_widget.show()
            end = time.time()
            #cb.remove()

            #end = time.time()
            print(str(self.count) + ' Elapsed time: ' + str(end - start))

            #self._ui.lineEdit_2.setText("Camera Type: " + self.camera.GetDeviceInfo().GetModelName())
            #self._ui.lineEdit_3.setText(
            #    "" + "{:.1f}".format(self.camera.AcquisitionFrameRateAbs.SetValue()))
            #self._ui.lineEdit.setText(
            #    "" + "{:.3f}".format(self.camera.ExposureTimeAbs.SetValue()))
            grab_result.Release()


            # For load data:
            # tif = TiffFile(data_path)
            # output = tifffile.imread(data_path, key = range(0,len(tif.pages),1))
            if self._ui.checkBox.isChecked():
                tifffile.imwrite(self.project_path + '\\' + self.current_tiff_file,np.array(image_orig,'uint16'),
                                 bigtiff=True,shaped=True,append=True,dtype='uint16',
                                 metadata = {'time_stamp':time.time()})




