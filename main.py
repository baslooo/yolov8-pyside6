from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, LOGGER, SETTINGS, callbacks, ops
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from ultralytics.utils.torch_utils import smart_inference_mode
from ultralytics.utils.files import increment_path
from ultralytics.utils.checks import check_imshow
from ultralytics.cfg import get_cfg
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu
from PySide6.QtGui import QImage, QPixmap, QColor
from PySide6.QtCore import QTimer, QThread, Signal, QObject, QPoint, Qt
from ui.CustomMessageBox import MessageBox
from ui.home import Ui_MainWindow
from ultralytics import YOLO
from ultralytics.data import load_inference_source
from ultralytics.utils.checks import check_imgsz, check_imshow
from UIFunctions import *
from collections import defaultdict
from pathlib import Path
from utils.capnums import Camera
from utils.rtsp_win import Window
import numpy as np
import time
import json
import torch
import sys
import cv2
import os


class YoloPredictor(BasePredictor, QObject):
    yolo2main_pre_img = Signal(np.ndarray)   # raw image signal
    yolo2main_res_img = Signal(np.ndarray)   # test result signal
    yolo2main_status_msg = Signal(str)       # Detecting/pausing/stopping/testing complete/error reporting signal
    yolo2main_fps = Signal(str)              # fps
    yolo2main_labels = Signal(dict)          # Detected target results (number of each category)
    yolo2main_progress = Signal(int)         # Completeness
    yolo2main_class_num = Signal(int)        # Number of categories detected
    yolo2main_target_num = Signal(int)       # Targets detected

    def __init__(self, cfg=DEFAULT_CFG, overrides=None): 
        super(YoloPredictor, self).__init__() 
        QObject.__init__(self)

        self.args = get_cfg(cfg, overrides)
        project = self.args.project or Path(SETTINGS['runs_dir']) / self.args.task
        name = f'{self.args.mode}'
        self.save_dir = increment_path(Path(project) / name, exist_ok=self.args.exist_ok)
        self.done_warmup = False
        if self.args.show:
            self.args.show = check_imshow(warn=True)

        # GUI args
        self.used_model_name = None      # The detection model name to use
        self.new_model_name = None       # Models that change in real time
        self.source = ''                 # input source
        self.stop_dtc = False            # Termination detection
        self.continue_dtc = True         # pause   
        self.save_res = False            # Save test results
        self.save_txt = False            # save label(txt) file
        self.iou_thres = 0.45            # iou
        self.conf_thres = 0.25           # conf
        self.speed_thres = 10            # delay, ms
        self.labels_dict = {}            # return a dictionary of results
        self.progress_value = 0          # progress bar
    

        # Usable if setup is done
        self.model = None
        self.data = self.args.data  # data_dict
        self.imgsz = None
        # self.device = None
        self.dataset = None
        self.vid_path, self.vid_writer = None, None
        self.annotator = None
        self.data_path = None
        self.source_type = None
        self.batch = None
        self.yolomodel =None
        self.callbacks = defaultdict(list, callbacks.default_callbacks)  # add callbacks
        callbacks.add_integration_callbacks(self)

    # main for detect
    @smart_inference_mode()
    def run(self):
        try:
            if self.args.verbose:
                LOGGER.info('')

            # set model    
            self.yolo2main_status_msg.emit('Loding Model...')

            if not self.yolomodel or self.used_model_name != self.new_model_name:
                self.yolomodel = YOLO(self.new_model_name)
                self.used_model_name = self.new_model_name
                custom = {'conf': self.conf_thres,'iou': self.iou_thres,'save': False}  # method defaults
                args = {**self.yolomodel.overrides, **custom, 'mode': 'predict'}  # highest priority args on the right
                self.predictor = (self.yolomodel._smart_load('predictor'))(overrides=args,
                                                                           _callbacks=self.callbacks)
                self.predictor.setup_model(model=self.yolomodel.model, verbose=False)
            self.predictor.setup_source(self.source if self.source is not None else self.predictor.args.source)

            if self.predictor.args.save or self.predictor.args.save_txt:
                (self.predictor.save_dir / 'labels' if self.predictor.args.save_txt else self.predictor.save_dir).mkdir(parents=True, exist_ok=True)
            # Warmup model
            if not self.predictor.done_warmup:
                self.predictor.model.warmup(imgsz=(1 if self.predictor.model.pt or self.predictor.model.triton else self.predictor.dataset.bs, 3, *self.predictor.imgsz))
                self.predictor.done_warmup = True

            self.seen, self.windows, self.batch, self.dt = 0, [], None, (ops.Profile(), ops.Profile(), ops.Profile())

            self.dataset = self.predictor.dataset
            count = 0                       # run location frame
            start_time = time.time()        # used to calculate the frame rate
            batch = iter(self.dataset)
            while True:
                # Termination detection
                if self.stop_dtc:
                    if isinstance(self.vid_writer[-1], cv2.VideoWriter):
                        self.vid_writer[-1].release()  # release final video writer
                    self.yolo2main_status_msg.emit('Detection terminated!')
                    break
                
                # Change the model midway
                if self.used_model_name != self.new_model_name:
                    self.yolo2main_status_msg.emit('Change Model...')
                    self.yolomodel = YOLO(self.new_model_name)
                    self.used_model_name = self.new_model_name
                    custom = {'conf': self.conf_thres, 'iou': self.iou_thres, 'save': False}  # method defaults
                    args = {**self.yolomodel.overrides, **custom,
                            'mode': 'predict'}  # highest priority args on the right
                    self.predictor = (self.yolomodel._smart_load('predictor'))(overrides=args,
                                                                               _callbacks=self.callbacks)
                    self.predictor.setup_model(model=self.yolomodel.model, verbose=False)
                    self.predictor.setup_source(self.source if self.source is not None else self.predictor.args.source)
                    self.dataset = self.predictor.dataset
                    batch = iter(self.dataset)

                # pause switch
                if self.continue_dtc:
                    # time.sleep(0.001)
                    self.yolo2main_status_msg.emit('Detecting...')
                    batch = next(self.dataset)  # next data


                    self.predictor.batch = batch

                    # path, im, im0s, vid_cap, s = batch
                    path, im0s, vid_cap, s = batch

                    count += 1              # frame count +1

                    # all_count = 1
                    # results =  self.yolomodel(self.source,stream= True)

                    if vid_cap:
                        all_count = vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)   # total frames
                    elif self.predictor.source_type.webcam:
                        all_count = -1
                    else :
                        all_count = 1
                    self.progress_value = int(count/all_count*1000)         # progress bar(0~1000)
                    if  count % 5 == 0 and count >= 5:                     # Calculate the frame rate every 5 frames
                        self.yolo2main_fps.emit(str(int(5/(time.time()-start_time))))
                        start_time = time.time()
                    # # preprocess
                    with self.dt[0]:
                        im = self.predictor.preprocess(im0s)
                        if len(im.shape) == 3:
                            im = im[None]  # expand for batch dim

                    # inference
                    with self.dt[1]:
                        visualize = increment_path(self.predictor.save_dir / Path(self.predictor.batch[0][0]).stem,
                                                   mkdir=True) if self.predictor.args.visualize and (
                            not self.predictor.source_type.tensor) else False
                        preds = self.predictor.model(im, augment=self.predictor.args.augment, visualize=visualize)
                    # postprocess
                    with self.dt[2]:
                        self.results = self.predictor.postprocess(preds, im, im0s)

                    # visualize, save, write results  
                    n = len(im0s)     # To be improved: support multiple img
                    for i in range(n):
                        self.results[i].speed = {
                            'preprocess': self.dt[0].dt * 1E3 / n,
                            'inference': self.dt[1].dt * 1E3 / n,
                            'postprocess': self.dt[2].dt * 1E3 / n}
                        p, im0 = path[i], None if self.predictor.source_type.tensor else im0s[i].copy()
                        p = Path(p)     # the source dir
                        label_str = self.write_results(i, self.results, (p, im, im0))   # labels   /// original :s +=
                    #
                        # labels and nums dict
                        class_nums = 0
                        target_nums = 0
                        self.labels_dict = {}
                        if 'no detections' in label_str:
                            pass
                        else:
                            for ii in label_str.split(',')[:-1]:
                                nums, label_name = ii.split('~')

                                self.labels_dict[label_name] = int(nums)
                                target_nums += int(nums)
                                class_nums += 1

                        # save img or video result
                        if self.save_res:
                            self.save_preds(vid_cap, i, str(self.predictor.save_dir / p.name))

                        # Send test results
                        # self.yolo2main_res_img.emit(im0) # after detection
                        self.yolo2main_res_img.emit(self.plotted_img) # after detection

                        self.yolo2main_pre_img.emit(im0s if isinstance(im0s, np.ndarray) else im0s[0])   # Before testing
                        self.yolo2main_labels.emit(self.labels_dict)        # webcam need to change the def write_results
                        self.yolo2main_class_num.emit(class_nums)
                        self.yolo2main_target_num.emit(target_nums)

                        if self.speed_thres != 0:
                            time.sleep(self.speed_thres/1000)   # delay , ms

                    self.yolo2main_progress.emit(self.progress_value)   # progress bar

                # Detection completed
                if count  == all_count:
                    if isinstance(self.predictor.vid_writer[-1], cv2.VideoWriter):
                        self.predictor.vid_writer[-1].release()  # release final video writer
                    self.yolo2main_status_msg.emit('Detection completed')
                    break

        except Exception as e:
            pass
            self.yolo2main_status_msg.emit('%s' % e)


    def get_annotator(self, img):
        return Annotator(img, line_width=None, example=str(self.predictor.model.names))



    def write_results(self, idx, results, batch):
        """Write inference results to a file or directory."""
        p, im, im0 = batch
        log_string = ''
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        if self.predictor.source_type.webcam or self.predictor.source_type.from_img or self.predictor.source_type.tensor:  # batch_size >= 1
            # log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)
        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + (
            '' if self.dataset.mode == 'image' else f'_{frame}')
        result = results[idx]

        det = results[idx].boxes  # TODO: make boxes inherit from tensors

        if len(det) == 0:
            return f'{log_string}(no detections), ' # if no, send this~~

        for c in det.cls.unique():
            n = (det.cls == c).sum()  # detections per class
            log_string += f"{n}~{self.predictor.model.names[int(c)]},"   #   {'s' * (n > 1)}, "   # don't add 's'

        self.annotator = self.get_annotator(im0)

        if self.args.save or self.args.show:  # Add bbox to image
            plot_args = {
                'line_width': self.predictor.args.line_width,
                'boxes': self.predictor.args.boxes,
                'conf': self.predictor.args.show_conf,
                'labels': self.predictor.args.show_labels}
            if not self.args.retina_masks:
                plot_args['im_gpu'] = im[idx]
            self.plotted_img = result.plot(**plot_args)
        # Write
        if self.args.save_txt:
            result.save_txt(f'{self.txt_path}.txt', save_conf=self.args.save_conf)
        if self.args.save_crop:
            result.save_crop(save_dir=self.save_dir / 'crops',
                             file_name=self.data_path.stem + ('' if self.dataset.mode == 'image' else f'_{frame}'))

        return log_string


class MainWindow(QMainWindow, Ui_MainWindow):
    main2yolo_begin_sgl = Signal()  # The main window sends an execution signal to the yolo instance
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        # basic interface
        self.setupUi(self)
        self.setAttribute(Qt.WA_TranslucentBackground)  # rounded transparent
        self.setWindowFlags(Qt.FramelessWindowHint)  # Set window flag: hide window borders
        UIFuncitons.uiDefinitions(self)
        # Show module shadows
        UIFuncitons.shadow_style(self, self.Class_QF, QColor(162,129,247))
        UIFuncitons.shadow_style(self, self.Target_QF, QColor(251, 157, 139))
        UIFuncitons.shadow_style(self, self.Fps_QF, QColor(170, 128, 213))
        UIFuncitons.shadow_style(self, self.Model_QF, QColor(64, 186, 193))
        


        # read model folder
        self.pt_list = os.listdir('./models')
        self.pt_list = [file for file in self.pt_list if file.endswith('.pt')]
        self.pt_list.sort(key=lambda x: os.path.getsize('./models/' + x))   # sort by file size
        self.model_box.clear()
        self.model_box.addItems(self.pt_list)
        self.Qtimer_ModelBox = QTimer(self)     # Timer: Monitor model file changes every 2 seconds
        self.Qtimer_ModelBox.timeout.connect(self.ModelBoxRefre)
        self.Qtimer_ModelBox.start(2000)

        # Yolo-v8 thread
        # self.yolo_predict = YOLO()
        self.yolo_predict = YoloPredictor()                           # Create a Yolo instance
        self.select_model = self.model_box.currentText()                   # default model
        self.yolo_predict.new_model_name = "./models/%s" % self.select_model  
        self.yolo_thread = QThread()                                  # Create yolo thread
        self.yolo_predict.yolo2main_pre_img.connect(lambda x: self.show_image(x, self.pre_video)) 
        self.yolo_predict.yolo2main_res_img.connect(lambda x: self.show_image(x, self.res_video))
        self.yolo_predict.yolo2main_status_msg.connect(lambda x: self.show_status(x))             
        self.yolo_predict.yolo2main_fps.connect(lambda x: self.fps_label.setText(x))              
        # self.yolo_predict.yolo2main_labels.connect(self.show_labels)                            
        self.yolo_predict.yolo2main_class_num.connect(lambda x:self.Class_num.setText(str(x)))         
        self.yolo_predict.yolo2main_target_num.connect(lambda x:self.Target_num.setText(str(x)))       
        self.yolo_predict.yolo2main_progress.connect(lambda x: self.progress_bar.setValue(x))     
        self.main2yolo_begin_sgl.connect(self.yolo_predict.run)     
        self.yolo_predict.moveToThread(self.yolo_thread)              

        # Model parameters
        self.model_box.currentTextChanged.connect(self.change_model)     
        self.iou_spinbox.valueChanged.connect(lambda x:self.change_val(x, 'iou_spinbox'))    # iou box
        self.iou_slider.valueChanged.connect(lambda x:self.change_val(x, 'iou_slider'))      # iou scroll bar
        self.conf_spinbox.valueChanged.connect(lambda x:self.change_val(x, 'conf_spinbox'))  # conf box
        self.conf_slider.valueChanged.connect(lambda x:self.change_val(x, 'conf_slider'))    # conf scroll bar
        self.speed_spinbox.valueChanged.connect(lambda x:self.change_val(x, 'speed_spinbox'))# speed box
        self.speed_slider.valueChanged.connect(lambda x:self.change_val(x, 'speed_slider'))  # speed scroll bar

        # Prompt window initialization
        self.Class_num.setText('--')
        self.Target_num.setText('--')
        self.fps_label.setText('--')
        self.Model_name.setText(self.select_model)
        
        # Select detection source
        self.src_file_button.clicked.connect(self.open_src_file)  # select local file
        self.src_cam_button.clicked.connect(self.chose_cam)#chose_cam
        # self.src_rtsp_button.clicked.connect(self.show_status("The function has not yet been implemented."))#chose_rtsp

        # start testing button
        self.run_button.clicked.connect(self.run_or_continue)   # pause/start
        self.stop_button.clicked.connect(self.stop)             # termination

        # Other function buttons
        self.save_res_button.toggled.connect(self.is_save_res)  # save image option
        self.save_txt_button.toggled.connect(self.is_save_txt)  # Save label option
        self.ToggleBotton.clicked.connect(lambda: UIFuncitons.toggleMenu(self, True))   # left navigation button
        self.settings_button.clicked.connect(lambda: UIFuncitons.settingBox(self, True))   # top right settings button
        
        # initialization
        self.load_config()

    # The main window displays the original image and detection results
    @staticmethod
    def show_image(img_src, label):
        try:
            ih, iw, _ = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()
            # keep the original data ratio
            if iw/w > ih/h:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))

            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print(repr(e))

    # Control start/pause
    def run_or_continue(self):
        if self.yolo_predict.source == '':
            self.show_status('Please select a video source before starting detection...')
            self.run_button.setChecked(False)
        else:
            self.yolo_predict.stop_dtc = False
            if self.run_button.isChecked():
                self.run_button.setChecked(True)    # start button
                self.save_txt_button.setEnabled(False)  # It is forbidden to check and save after starting the detection
                self.save_res_button.setEnabled(False)
                self.show_status('Detecting...')           
                self.yolo_predict.continue_dtc = True   # Control whether Yolo is paused
                if not self.yolo_thread.isRunning():
                    self.yolo_thread.start()
                    self.main2yolo_begin_sgl.emit()

            else:
                self.yolo_predict.continue_dtc = False
                self.show_status("Pause...")
                self.run_button.setChecked(False)    # start button

    # bottom status bar information
    def show_status(self, msg):
        self.status_bar.setText(msg)
        if msg == 'Detection completed' or msg == '检测完成':
            self.save_res_button.setEnabled(True)
            self.save_txt_button.setEnabled(True)
            self.run_button.setChecked(False)    
            self.progress_bar.setValue(0)
            if self.yolo_thread.isRunning():
                self.yolo_thread.quit()         # end process
        elif msg == 'Detection terminated!' or msg == '检测终止':
            self.save_res_button.setEnabled(True)
            self.save_txt_button.setEnabled(True)
            self.run_button.setChecked(False)    
            self.progress_bar.setValue(0)
            if self.yolo_thread.isRunning():
                self.yolo_thread.quit()         # end process
            self.pre_video.clear()           # clear image display  
            self.res_video.clear()          
            self.Class_num.setText('--')
            self.Target_num.setText('--')
            self.fps_label.setText('--')

    # select local file
    def open_src_file(self):
        config_file = 'config/fold.json'    
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        open_fold = config['open_fold']     
        if not os.path.exists(open_fold):
            open_fold = os.getcwd()
        name, _ = QFileDialog.getOpenFileName(self, 'Video/image', open_fold, "Pic File(*.mp4 *.mkv *.avi *.flv *.jpg *.png)")
        if name:
            self.yolo_predict.source = name
            self.show_status('Load File：{}'.format(os.path.basename(name))) 
            config['open_fold'] = os.path.dirname(name)
            config_json = json.dumps(config, ensure_ascii=False, indent=2)  
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)
            self.stop()             

    # Select camera source----  have one bug
    def chose_cam(self):
        try:
            self.stop()
            MessageBox(
                self.close_button, title='Note', text='loading camera...', time=2000, auto=True).exec()
            # get the number of local cameras
            _, cams = Camera().get_cam_num()
            popMenu = QMenu()
            popMenu.setFixedWidth(self.src_cam_button.width())
            popMenu.setStyleSheet('''
                                            QMenu {
                                            font-size: 16px;
                                            font-family: "Microsoft YaHei UI";
                                            font-weight: light;
                                            color:white;
                                            padding-left: 5px;
                                            padding-right: 5px;
                                            padding-top: 4px;
                                            padding-bottom: 4px;
                                            border-style: solid;
                                            border-width: 0px;
                                            border-color: rgba(255, 255, 255, 255);
                                            border-radius: 3px;
                                            background-color: rgba(200, 200, 200,50);}
                                            ''')

            for cam in cams:
                # cap = cv2.VideoCapture(cam)
                # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 设置分辨率
                # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                exec("action_%s = QAction('%s')" % (cam, cam))
                exec("popMenu.addAction(action_%s)" % cam)

            x = self.src_cam_button.mapToGlobal(self.src_cam_button.pos()).x()     
            y = self.src_cam_button.mapToGlobal(self.src_cam_button.pos()).y()     
            y = y + self.src_cam_button.frameGeometry().height()
            pos = QPoint(x, y)
            action = popMenu.exec(pos)
            if action:
                self.yolo_predict.source = action.text()

                self.show_status('Loading camera：{}'.format(action.text()))

        except Exception as e:
            self.show_status('%s' % e)

    # select network source
    def chose_rtsp(self):
        self.rtsp_window = Window()
        config_file = 'config/ip.json'
        if not os.path.exists(config_file):
            ip = "rtsp://admin:admin888@192.168.1.2:555"
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            ip = config['ip']
        self.rtsp_window.rtspEdit.setText(ip)
        self.rtsp_window.show()
        self.rtsp_window.rtspButton.clicked.connect(lambda: self.load_rtsp(self.rtsp_window.rtspEdit.text()))
    
    # load network sources
    def load_rtsp(self, ip):
        try:
            self.stop()
            MessageBox(
                self.close_button, title='提示', text='加载 rtsp...', time=1000, auto=True).exec()
            self.yolo_predict.source = ip
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open('config/ip.json', 'w', encoding='utf-8') as f:
                f.write(new_json)
            self.show_status('Loading rtsp：{}'.format(ip))
            self.rtsp_window.close()
        except Exception as e:
            self.show_status('%s' % e)

    # Save test result button--picture/video
    def is_save_res(self):
        if self.save_res_button.checkState() == Qt.CheckState.Unchecked:
            self.show_status('NOTE: Run image results are not saved.')
            self.yolo_predict.save_res = False
        elif self.save_res_button.checkState() == Qt.CheckState.Checked:
            self.show_status('NOTE: Run image results will be saved.')
            self.yolo_predict.save_res = True
    
    # Save test result button -- label (txt)
    def is_save_txt(self):
        if self.save_txt_button.checkState() == Qt.CheckState.Unchecked:
            self.show_status('NOTE: Labels results are not saved.')
            self.yolo_predict.save_txt = False
        elif self.save_txt_button.checkState() == Qt.CheckState.Checked:
            self.show_status('NOTE: Labels results will be saved.')
            self.yolo_predict.save_txt = True

    # Configuration initialization  ~~~wait to change~~~
    def load_config(self):
        config_file = 'config/setting.json'
        if not os.path.exists(config_file):
            iou = 0.26
            conf = 0.33   
            rate = 10
            save_res = 0   
            save_txt = 0    
            new_config = {"iou": iou,
                          "conf": conf,
                          "rate": rate,
                          "save_res": save_res,
                          "save_txt": save_txt
                          }
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            if len(config) != 5:
                iou = 0.26
                conf = 0.33
                rate = 10
                save_res = 0
                save_txt = 0
            else:
                iou = config['iou']
                conf = config['conf']
                rate = config['rate']
                save_res = config['save_res']
                save_txt = config['save_txt']
        self.save_res_button.setCheckState(Qt.CheckState(save_res))
        self.yolo_predict.save_res = (False if save_res==0 else True )
        self.save_txt_button.setCheckState(Qt.CheckState(save_txt)) 
        self.yolo_predict.save_txt = (False if save_txt==0 else True )
        self.run_button.setChecked(False)  
        self.show_status("Welcome~")

    # Terminate button and associated state
    def stop(self):
        if self.yolo_thread.isRunning():
            self.yolo_thread.quit()         # end thread
        self.yolo_predict.stop_dtc = True
        self.run_button.setChecked(False)    # start key recovery
        self.save_res_button.setEnabled(True)   # Ability to use the save button
        self.save_txt_button.setEnabled(True)   # Ability to use the save button
        self.pre_video.clear()           # clear image display
        self.res_video.clear()           # clear image display
        self.progress_bar.setValue(0)
        self.Class_num.setText('--')
        self.Target_num.setText('--')
        self.fps_label.setText('--')

    # Change detection parameters
    def change_val(self, x, flag):
        if flag == 'iou_spinbox':
            self.iou_slider.setValue(int(x*100))    # The box value changes, changing the slider
        elif flag == 'iou_slider':
            self.iou_spinbox.setValue(x/100)        # The slider value changes, changing the box
            self.show_status('IOU Threshold: %s' % str(x/100))
            self.yolo_predict.iou_thres = x/100
        elif flag == 'conf_spinbox':
            self.conf_slider.setValue(int(x*100))
        elif flag == 'conf_slider':
            self.conf_spinbox.setValue(x/100)
            self.show_status('Conf Threshold: %s' % str(x/100))
            self.yolo_predict.conf_thres = x/100
        elif flag == 'speed_spinbox':
            self.speed_slider.setValue(x)
        elif flag == 'speed_slider':
            self.speed_spinbox.setValue(x)
            self.show_status('Delay: %s ms' % str(x))
            self.yolo_predict.speed_thres = x  # ms
            
    # change model
    def change_model(self,x):
        self.select_model = self.model_box.currentText()
        self.yolo_predict.new_model_name = "./models/%s" % self.select_model
        self.show_status('Change Model：%s' % self.select_model)
        self.Model_name.setText(self.select_model)

    # label result
    # def show_labels(self, labels_dic):
    #     try:
    #         self.result_label.clear()
    #         labels_dic = sorted(labels_dic.items(), key=lambda x: x[1], reverse=True)
    #         labels_dic = [i for i in labels_dic if i[1]>0]
    #         result = [' '+str(i[0]) + '：' + str(i[1]) for i in labels_dic]
    #         self.result_label.addItems(result)
    #     except Exception as e:
    #         self.show_status(e)

    # Cycle monitoring model file changes
    def ModelBoxRefre(self):
        pt_list = os.listdir('./models')
        pt_list = [file for file in pt_list if file.endswith('.pt')]
        pt_list.sort(key=lambda x: os.path.getsize('./models/' + x))
        # It must be sorted before comparing, otherwise the list will be refreshed all the time
        if pt_list != self.pt_list:
            self.pt_list = pt_list
            self.model_box.clear()
            self.model_box.addItems(self.pt_list)

    # Get the mouse position (used to hold down the title bar and drag the window)
    def mousePressEvent(self, event):
        p = event.globalPosition()
        globalPos = p.toPoint()
        self.dragPos = globalPos

    # Optimize the adjustment when dragging the bottom and right edges of the window size
    def resizeEvent(self, event):
        # Update Size Grips
        UIFuncitons.resize_grips(self)

    # Exit Exit thread, save settings
    def closeEvent(self, event):
        config_file = 'config/setting.json'
        config = dict()
        config['iou'] = self.iou_spinbox.value()
        config['conf'] = self.conf_spinbox.value()
        config['rate'] = self.speed_spinbox.value()
        config['save_res'] = (0 if self.save_res_button.checkState()==Qt.Unchecked else 2)
        config['save_txt'] = (0 if self.save_txt_button.checkState()==Qt.Unchecked else 2)
        config_json = json.dumps(config, ensure_ascii=False, indent=2)
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_json)
        # Exit the process before closing
        if self.yolo_thread.isRunning():
            self.yolo_predict.stop_dtc = True
            self.yolo_thread.quit()
            MessageBox(
                self.close_button, title='Note', text='Exiting, please wait...', time=3000, auto=True).exec()
            sys.exit(0)
        else:
            sys.exit(0)

class LoadWebcam:  # for inference
    # YOLOv5 local webcam dataloader, i.e. `python detect.py --source 0`
    def __init__(self, pipe='0', img_size=640, stride=32):
        self.img_size = img_size
        self.stride = stride
        self.pipe = eval(pipe) if pipe.isnumeric() else pipe
        self.cap = cv2.VideoCapture(self.pipe)  # video capture object
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # set buffer size

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord('q'):  # q to quit
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration

        # Read frame
        ret_val, img0 = self.cap.read()
        img0 = cv2.flip(img0, 1)  # flip left-right

        # Print
        assert ret_val, f'Camera Error {self.pipe}'
        img_path = 'webcam.jpg'
        s = f'webcam {self.count}: '

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return img_path, img, img0, self.cap

    def __len__(self):
        return 0

if __name__ == "__main__":
    app = QApplication(sys.argv)
    Home = MainWindow()
    Home.show()
    sys.exit(app.exec())  
