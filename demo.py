import platform
import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from PyQt5.QtWidgets import QApplication
import sys
from PyQt5.QtWidgets import QWidget
from PyQt5.Qt import QPixmap, QPainter, QPoint, QPaintEvent, QMouseEvent, QPen,\
    QColor, QSize
from PyQt5.QtCore import Qt
from PyQt5.Qt import QWidget, QColor, QPixmap, QIcon, QSize, QCheckBox
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QPushButton, QSplitter,\
    QComboBox, QLabel, QSpinBox, QFileDialog
    
from simple_lama import SimpleLamaTest

from PIL import Image

from transformers import Mask2FormerForUniversalSegmentation
from transformers import Mask2FormerImageProcessor
import numpy as np
import torch
from scipy.ndimage import convolve

#洪水扩充
def update_grid(grid):
    # 定义卷积核
    kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]])
    # 使用卷积操作
    convolved = convolve(grid, kernel, mode='constant', cval=0)
    # 返回更新后的值
    return np.logical_or((grid == 1), (convolved >= 1))

def update_multiple(grid, k):
    for _ in range(k):
        grid = update_grid(grid)
    return grid.astype('uint8')

# 示例用法
#grid = np.array([[0, 0, 0, 0, 0],
#                 [1, 0, 0, 0, 0],
#                 [0, 0, 0, 1, 0],
#                 [0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0]])
#
#k = 1
#result = update_multiple(grid, k)
#print(result)
#[[1 0 0 0 0]
# [1 1 0 1 0]
# [1 0 1 1 1]
# [0 0 0 1 0]
# [0 0 0 0 0]]

def convert_pil_to_qimage(pil_image):
    if pil_image.mode == 'RGB':
        image_format = QImage.Format_RGB888
    elif pil_image.mode == 'RGBA':
        image_format = QImage.Format_RGBA8888
    else:
        raise ValueError("Unsupported image mode: {}".format(pil_image.mode))
    
    image_data = pil_image.tobytes('raw', pil_image.mode)
    qimage = QImage(image_data, pil_image.width, pil_image.height, image_format)
    return qimage

def main():
    app = QApplication(sys.argv)

    mainWidget = MainWidget() #新建一个主界面
    mainWidget.show()    #显示主界面

    exit(app.exec_()) #进入消息循环


class PaintBoard(QWidget):

    def __init__(self, image_path, width, height, Parent=None):
        '''
        Constructor
        '''
        super().__init__(Parent)

        self.__InitData(image_path, width, height)  # 先初始化数据，再初始化界面
        self.__InitView()
        self.setWindowTitle("画笔")

    def __InitData(self, image_path, width, height):

        self.__size = QSize(width, height)
        
        # 新建QLabel作为背景
        self.background = QLabel(self)
        self.background.setFixedSize(self.__size)
        #img = Image.open(r"D:\Program Files\fdu\py\1.jpg")
        #resize = transforms.Resize([460,480])
        #img = resize(img)
        #img.save(r"D:\Program Files\fdu\py\1.jpg")
        self.background.setPixmap(QPixmap(image_path))
        
        # 新建QPixmap作为画板，尺寸为__size
        
        self.board = QPixmap(self.__size)
        self.board.fill(QColor(0, 0, 0, 128))  # 用透明色填充画板
        
        self.board_show = QLabel(self)
        self.board_show.setFixedSize(self.__size)
        self.board_show.setPixmap(self.board)
        
        

        self.IsEmpty = True  # 默认为空画板
        self.EraserMode = False  # 默认为禁用橡皮擦模式

        self.__lastPos = QPoint(0, 0)  # 上一次鼠标位置
        self.__currentPos = QPoint(0, 0)  # 当前的鼠标位置

        self.__painter = QPainter()  # 新建绘图工具
        

        self.__thickness = 10  # 默认画笔粗细为10px
        self.__penColor = QColor(255,0,0,128)  # 设置默认画笔颜色为红色
        self.__colorList = QColor.colorNames()  # 获取颜色列表
        

    def __InitView(self):
        # 设置界面的尺寸为__size
        self.setFixedSize(self.__size)
    

    def ChangePenThickness(self, thickness=10):
        # 改变画笔粗细
        self.__thickness = thickness

    def IsEmpty(self):
        # 返回画板是否为空
        return self.IsEmpty

    def GetContentAsQImage(self):
        # 获取画板内容（返回QImage）
        image = self.board.toImage()
        return image

    def paintEvent(self, paintEvent):
        # 绘图事件
        # 绘图时必须使用QPainter的实例，此处为__painter
        # 绘图在begin()函数与end()函数间进行
        # begin(param)的参数要指定绘图设备，即把图画在哪里
        # drawPixmap用于绘制QPixmap类型的对象
        self.__painter.begin(self)
        # 0,0为绘图的左上角起点的坐标，__board即要绘制的图
        self.__painter.drawPixmap(0, 0, self.board)
        self.__painter.end()

    def mousePressEvent(self, mouseEvent):
        # 鼠标按下时，获取鼠标的当前位置保存为上一次位置
        self.__currentPos = mouseEvent.pos()
        self.__lastPos = self.__currentPos

    def mouseMoveEvent(self, mouseEvent):
        # 鼠标移动时，更新当前位置，并在上一个位置和当前位置间画线
        self.__currentPos = mouseEvent.pos()
        self.__painter.begin(self.board)

        if self.EraserMode == False:
            # 非橡皮擦模式
            self.__painter.setCompositionMode(QPainter.CompositionMode_Source)
            #self.__painter.setOpacity(0.5)
            self.__painter.setPen(QPen(self.__penColor, self.__thickness, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))  # 设置画笔颜色，粗细
        else:
            # 橡皮擦模式下画笔为透明色
            self.__painter.setCompositionMode(QPainter.CompositionMode_Source)
            self.__painter.setPen(QPen(QColor(0, 0, 0, 128), self.__thickness, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))

        # 画线
        
        self.__painter.drawLine(self.__lastPos, self.__currentPos)
        self.__painter.end()
        self.__lastPos = self.__currentPos

        self.update()  # 更新显示
        
        
        self.board_show.setPixmap(self.board)


    def mouseReleaseEvent(self, mouseEvent):
        self.IsEmpty = False  # 画板不再为空


class MainWidget(QWidget):

    def __init__(self, Parent=None):
        '''
        Constructor
        '''
        super().__init__(Parent)

        self.__InitData()  # 先初始化数据，再初始化界面
        self.__InitView()

    def __InitData(self):
        '''
                  初始化成员变量
        '''
        self.__paintBoard = PaintBoard('', 0, 0, self)
        self.__outputBoard = PaintBoard('', 0, 0, self)
        # 获取颜色列表(字符串类型)
        self.__colorList = QColor.colorNames()
        self.inpainter = SimpleLamaTest(model_path="./model/big-lama.pt")
        self.segmentation_processor = Mask2FormerImageProcessor(ignore_index=255, reduce_labels=False, do_resize=False, do_rescale=False, do_normalize=False)
        self.segmentator = Mask2FormerForUniversalSegmentation.from_pretrained("./model/mask2former/")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.segmentator.to(self.device)
        self.segmentator.eval()
        
        self.input_image = None
        self.output_image = None

    def __InitView(self):
        '''
                  初始化界面
        '''
        self.setMinimumSize(1900, 1000)
        self.setWindowTitle("Demo")

        # 新建一个水平布局作为本窗体的主布局
        self.main_layout = QHBoxLayout(self)
        # 设置主布局内边距以及控件间距为10px
        self.main_layout.setSpacing(10)

        # 图像相关按键
        self.left_buttons = QVBoxLayout()
        self.left_buttons.setContentsMargins(10, 10, 10, 10)
        
        self.__btn_selectImage = QPushButton("导入图像")
        self.__btn_selectImage.setParent(self)  # 设置父对象为本界面
        self.__btn_selectImage.clicked.connect(self.open_image) 
        self.left_buttons.addWidget(self.__btn_selectImage)
        
        self.__btn_selectMask = QPushButton("导入Mask")
        self.__btn_selectMask.setParent(self)  # 设置父对象为本界面
        self.__btn_selectMask.clicked.connect(self.open_mask)
        self.left_buttons.addWidget(self.__btn_selectMask)

        self.__btn_generateMask = QPushButton("生成Mask")
        self.__btn_generateMask.setParent(self)  # 设置父对象为本界面
        self.__btn_generateMask.clicked.connect(self.generate_mask)
        self.left_buttons.addWidget(self.__btn_generateMask)
        
        self.__btn_saveMask = QPushButton("保存Mask")
        self.__btn_saveMask.setParent(self)
        self.__btn_saveMask.clicked.connect(self.on_btn_Save_Clicked)
        self.left_buttons.addWidget(self.__btn_saveMask)
                
        splitter = QSplitter(self)  # 占位符
        self.left_buttons.addWidget(splitter)
        
        
        
        self.__btn_callLama = QPushButton("生成修复图像")
        self.__btn_callLama.setParent(self)  # 设置父对象为本界面
        self.__btn_callLama.clicked.connect(self.call_lama)
        self.left_buttons.addWidget(self.__btn_callLama)
        self.main_layout.addLayout(self.left_buttons)

        # 输入图像
        self.left_layout = QVBoxLayout()
        self.left_layout.setContentsMargins(10, 10, 10, 10)
        splitter = QSplitter(self)  # 占位符
        self.left_layout.addWidget(splitter)
        self.__paintBoard_name = QLabel(self)
        self.__paintBoard_name.setText("输入图像&Mask")
        self.left_layout.addWidget(self.__paintBoard_name)
        self.left_layout.addWidget(self.__paintBoard)
        self.main_layout.addLayout(self.left_layout)
        
        # 输出图像
        self.mid_layout = QVBoxLayout()
        self.mid_layout.setContentsMargins(10, 10, 10, 10)
        splitter = QSplitter(self)  # 占位符
        self.mid_layout.addWidget(splitter)
        self.__paintBoard_name = QLabel(self)
        self.__paintBoard_name.setText("输出图像")
        self.mid_layout.addWidget(self.__paintBoard_name)
        self.mid_layout.addWidget(self.__outputBoard)
        self.main_layout.addLayout(self.mid_layout)

        # 新建垂直子布局用于放置按键
        self.sub_layout = QVBoxLayout()

        # 设置此子布局和内部控件的间距为10px
        self.sub_layout.setContentsMargins(10, 10, 10, 10)

        self.__btn_Quit = QPushButton("退出")
        self.__btn_Quit.setParent(self)  # 设置父对象为本界面
        self.__btn_Quit.clicked.connect(self.Quit)
        self.sub_layout.addWidget(self.__btn_Quit)

        self.__btn_Save = QPushButton("保存修复图像")
        self.__btn_Save.setParent(self)
        self.__btn_Save.clicked.connect(self.save_inpainting_image)
        self.sub_layout.addWidget(self.__btn_Save)

        splitter = QSplitter(self)  # 占位符
        self.sub_layout.addWidget(splitter)

        self.__btn_Clear = QPushButton("清空Mask")
        self.__btn_Clear.setParent(self)  # 设置父对象为本界面

        # 将按键按下信号与画板清空函数相关联
        self.__btn_Clear.clicked.connect(self.Clear)
        self.sub_layout.addWidget(self.__btn_Clear)

        
        self.__cbtn_Eraser = QCheckBox("使用橡皮擦")
        self.__cbtn_Eraser.setParent(self)
        self.__cbtn_Eraser.clicked.connect(self.on_cbtn_Eraser_clicked)
        self.sub_layout.addWidget(self.__cbtn_Eraser)

        

        self.__label_penThickness = QLabel(self)
        self.__label_penThickness.setText("画笔粗细")
        self.__label_penThickness.setFixedHeight(20)
        self.sub_layout.addWidget(self.__label_penThickness)

        self.__spinBox_penThickness = QSpinBox(self)
        self.__spinBox_penThickness.setMaximum(40)
        self.__spinBox_penThickness.setMinimum(2)
        self.__spinBox_penThickness.setValue(10)  # 默认粗细为10
        self.__spinBox_penThickness.setSingleStep(2)  # 最小变化值为2
        self.__spinBox_penThickness.valueChanged.connect(
            self.on_PenThicknessChange)  # 关联spinBox值变化信号和函数on_PenThicknessChange
        self.sub_layout.addWidget(self.__spinBox_penThickness)


        self.main_layout.addLayout(self.sub_layout)  # 将子布局加入主布局


    def open_image(self):
        """
        select image file and open it
        :return:
        """
        # img_name, _ = QFileDialog.getOpenFileName(self, "打开图片", "", "All Files(*);;*.jpg;;*.png")
        img_name, _ = QFileDialog.getOpenFileName(self, "Open Image File","","All Files(*);;*.jpg;;*.png;;*.jpeg")
        if img_name == "":
            print("cancel")
            return
        self.input_image = Image.open(img_name)
        w = self.input_image.size[0]
        h = self.input_image.size[1]
        index = self.left_layout.count()
        self.left_layout.itemAt(index-1).widget().deleteLater() # 删除之前的画板
        self.__paintBoard  = PaintBoard(img_name, w, h, self)
        scroll_area = QScrollArea() # 新建一个滚动区域包住画板
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumSize(QSize(900,600))
        scroll_area.setWidget(self.__paintBoard)
        self.left_layout.addWidget(scroll_area)

    def open_mask(self):
        if self.input_image == None:
            warning = QMessageBox.warning(self, "Warning", "Please import image first", QMessageBox.Yes)
            print(warning)
        else:

            img_name, _ = QFileDialog.getOpenFileName(self, "Open Image File","","*.png")
            if img_name == "":
                print("cancel")
                return
            self.__paintBoard.board = QPixmap(img_name)
            self.__paintBoard.update()
            self.__paintBoard.board_show.setPixmap(self.__paintBoard.board)
        

    def on_PenThicknessChange(self):
        penThickness = self.__spinBox_penThickness.value()
        self.__paintBoard.ChangePenThickness(penThickness)

    def on_btn_Save_Clicked(self):
        if self.input_image == None:
            warning = QMessageBox.warning(self, "Warning", "Please import image first", QMessageBox.Yes)
            print(warning)
            return
        savePath = QFileDialog.getSaveFileName(self, 'Save Your Paint', '.\\', '*.png')
        print(savePath)
        if savePath[0] == "":
            print("Save cancel")
            return
        image = self.__paintBoard.GetContentAsQImage()
        image.save(savePath[0])

    def on_cbtn_Eraser_clicked(self):
        if self.__cbtn_Eraser.isChecked():
            self.__paintBoard.EraserMode = True  # 进入橡皮擦模式
        else:
            self.__paintBoard.EraserMode = False  # 退出橡皮擦模式

    def Clear(self):
        # 清空Mask
        self.__paintBoard.board.fill(QColor(0, 0, 0, 128))
        self.__paintBoard.update()
        self.__paintBoard.board_show.setPixmap(self.__paintBoard.board)
        self.__paintBoard.IsEmpty = True
    
    def Quit(self):
        self.close()
        
        
    def generate_mask(self):
        if self.input_image == None:
            warning = QMessageBox.warning(self, "Warning", "Please import image first", QMessageBox.Yes)
            print(warning)
        else:
            image_array = np.array(self.input_image)
            batch = self.segmentation_processor(image_array, return_tensors="pt",)
            with torch.no_grad():
                outputs = self.segmentator(batch["pixel_values"].float().to(self.device))
            target_sizes = [(image_array.shape[0], image_array.shape[1])]
            predicted_segmentation_maps = self.segmentation_processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
            segmentation_map = predicted_segmentation_maps[0].cpu().numpy()
            segmentation_map = update_multiple(segmentation_map, 3)
            color_segmentation_map = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1], 3), dtype=np.uint8) # height, width, 3
            color_segmentation_map[segmentation_map == 1, :] = [0, 0, 255]
            # Convert to BGR
            ground_truth_color_seg = color_segmentation_map[..., ::-1]
            pil_image = Image.fromarray(ground_truth_color_seg.astype('uint8'))
            pil_image.putalpha(128) # 50% 透明度
            # 将 PIL.Image 转换为 QImage
            qimage = convert_pil_to_qimage(pil_image)
            self.__paintBoard.board = QPixmap.fromImage(qimage)
            self.__paintBoard.board_show.setPixmap(self.__paintBoard.board)

            
    def call_lama(self):
        if self.input_image == None:
            warning = QMessageBox.warning(self, "Warning", "Please import image first", QMessageBox.Yes)
            print(warning)
        else:
            index = self.mid_layout.count()
            self.mid_layout.itemAt(index-1).widget().deleteLater() # 删除之前的画板
            mask_rgba = self.__paintBoard.board.toImage()
            pil_mask_rgba = Image.fromqpixmap(mask_rgba)
            mask_rgba_array = np.array(pil_mask_rgba)
            mask_array = mask_rgba_array[:,:,0].reshape(mask_rgba_array.shape[0], mask_rgba_array.shape[1])
            mask = Image.fromarray(mask_array.astype('uint8'))
            self.output_image = self.inpainter(self.input_image, mask)
            
            self.__outputBoard  = QLabel(self)
            self.__outputBoard.setPixmap(QPixmap.fromImage(convert_pil_to_qimage(self.output_image)))
            scroll_area = QScrollArea() # 新建一个滚动区域包住画板
            scroll_area.setWidgetResizable(True)
            scroll_area.setMinimumSize(QSize(900,600))
            scroll_area.setWidget(self.__outputBoard)
            self.mid_layout.addWidget(scroll_area)
            
            
            
    def save_inpainting_image(self):
        if self.output_image == None:
            warning = QMessageBox.warning(self, "Warning", "Please call lama first", QMessageBox.Yes)
            print(warning)
        else:
            savePath = QFileDialog.getSaveFileName(self, 'Save Your Paint', '.\\', '*.jpg')
            print(savePath)
            if savePath[0] == "":
                print("Save cancel")
                return
            self.output_image.save(savePath[0])


if __name__ == '__main__':
    main()
