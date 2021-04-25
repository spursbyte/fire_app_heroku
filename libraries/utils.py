#------------------------------------------------------#
# Import librairies
#------------------------------------------------------#

import datetime
import hashlib
import os
import time
import urllib
import io
import cv2 as cv
import numpy as np
import pafy
import pandas as pd
import streamlit as st
import wget
import argparse

import youtube_dl
from imutils.video import FPS, FileVideoStream, WebcamVideoStream
from PIL import Image

import libraries.plugins as plugins

colorWhite = (255, 255, 255)
colorBlack = (0, 0, 0)
colorRed = (255, 0, 0)
colorGreen = (0, 255, 0)
colorBlue = (0, 0, 255)
fontFace = cv.FONT_HERSHEY_SIMPLEX
thickText = 1

#------------------------------------------------------#
# Classes definition
#------------------------------------------------------#


class GUI():
    """
    This class is dedicated to manage to user interface of the website. It contains methods to edit the sidebar for the selected application as well as the front page.
    """

    def __init__(self):

        self.list_of_apps = [
            'Empty',
            'Fire Detection from Video',
        
            'Fire Detection']
        self.guiParam = {}

    # ----------------------------------------------------------------

    def getGuiParameters(self):
        self.common_config()
        self.appDescription()
        return self.guiParam

    # ------------------------------------a----------------------------

    def common_config(self, title='Fire Detection Using Deep Learning '): #(Beta version :golf:)
        """
        User Interface Management: Sidebar
        """
        st.image("./media/coeai.png", width=120)

        st.title(title)

        st.sidebar.markdown("### :bulb: Settings")

        # Get the application type from the GUI
        self.appType = 'Image Applications'

        self.dataSource = st.sidebar.radio(
                'Please select the source for Fire Detection ', ['Video: Upload', 'Image: Upload'])


        # Get the application from the GUI
        self.selectedApp = st.sidebar.selectbox(
            'Chose an AI Application', self.list_of_apps)

        if self.selectedApp is 'Empty':
            st.sidebar.warning('Select an application from the list')
        
        

        

        # Update the dictionnary
        self.guiParam.update(
            dict(selectedApp=self.selectedApp,
                 appType=self.appType,
                 dataSource=self.dataSource,
                 ))

    # -------------------------------------------------------------------------

    def appDescription(self):

        st.header(' :computer: Application: {}'.format(self.selectedApp))

        
        if self.selectedApp == 'Fire Detection':
            st.info(
                'This application performs fire detection on Images using Deep Learning models. ')
            self.sidebarFireDetection()
        elif self.selectedApp == 'Fire Detection from Video':
            st.info(
                'This application performs fire detection on Video Sequences using Advanced Deep Learning models. ')

        else:
            st.info(
                'To start using Fire Detection Application, you must first select an Application from the sidebar menu other than Empty. \n Below is a visual demo of the working model.')
            video_file = open('firedet_demo.webm', 'rb')
            video_bytes = video_file.read()
            
            st.video(video_bytes)
    # --------------------------------------------------------------------------
    def sidebarEmpty(self):
        pass
    # --------------------------------------------------------------------------

    

    def sidebarFireDetection(self):

        # st.sidebar.markdown("### :arrow_right: Model")
        #------------------------------------------------------#
        model = st.sidebar.selectbox(
            label='Select the model',
            options=['Darknet-YOLOv3-tiny'])

        # st.sidebar.markdown("### :arrow_right: Model Parameters")
        #------------------------------------------------------#
        confThresh = st.sidebar.slider(
            'Confidence', value=0.5, min_value=0.0, max_value=1.0)
        nmsThresh = st.sidebar.slider(
            'Non-maximum suppression', value=0.30, min_value=0.0, max_value=1.00, step=0.05)

        self.guiParam.update(dict(confThresh=confThresh,
                                  nmsThresh=nmsThresh,
                                  model=model))


# ------------------------------------------------------------------
# ------------------------------------------------------------------


class AppManager:
    """
    This is a master class
    """

    def __init__(self, guiParam):
        self.guiParam = guiParam
        self.selectedApp = guiParam['selectedApp']

        self.model = guiParam['model']
        self.objApp = self.setupApp()

    # -----------------------------------------------------

    def setupApp(self):
        """
        #
        """

        
        

        if self.selectedApp == 'Fire Detection':
            @st.cache(allow_output_mutation=True)
            def getClasses(classesFile):
                """
                # Load names of classes
                """
                classes = None
                with open(classesFile, 'rt') as f:
                    classes = f.read().rstrip('\n').split('\n')
                return classes

            labels = 'models/DarkNet/fire_detection/classes.names'
            self.paramYoloTinyFire = dict(labels=labels,
                                          modelCfg='models/DarkNet/fire_detection/yolov3-custom.cfg',
                                          modelWeights="models/DarkNet/fire_detection/yolov3-custom_10000.weights",
                                          confThresh=self.guiParam['confThresh'],
                                          nmsThresh=self.guiParam['nmsThresh'],
                                          colors=np.tile(colorBlue, (len(getClasses(labels)), 1)).tolist())

            self.objApp = plugins.Object_Detection_YOLO(self.paramYoloTinyFire)

        # -----------------------------------------------------

        else:
            raise Exception(
                '[Error] Please select one of the listed application')

        return self.objApp

    # -----------------------------------------------------
    # -----------------------------------------------------

    def process(self, frame, motion_state):
        """
        # return a tuple: (bboxed_frame, output)
        """
        bboxed_frame, output = self.objApp.run(frame, motion_state)

        return bboxed_frame, output

# ------------------------------------------------------------------
# ------------------------------------------------------------------


class DataManager:
    """
    """

    def __init__(self, guiParam):
        self.guiParam = guiParam

        
        self.image = None
        self.data = None

  #################################################################
  #################################################################

    def load_image_source(self):
        """
        """

        if self.guiParam["dataSource"] == 'Image: Upload':

            @st.cache(allow_output_mutation=True)
            def load_image_from_upload(file):
                tmp = np.fromstring(file.read(), np.uint8)
                return cv.imdecode(tmp, 1)

            file_path = st.file_uploader(
                'Upload an image', type=['png', 'jpg'])

            if file_path is not None:
                self.image = load_image_from_upload(file_path)
            
            #--------------------------------------------#
            #--------------------------------------------#
            return self.image

        elif self.guiParam["dataSource"] == 'Video: Upload':
            parser = argparse.ArgumentParser(add_help=False)
            parser.add_argument("--image", default='custom/traintest/17.jpg', help="image for prediction")
            parser.add_argument("--config", default='models/DarkNet/fire_detection/yolov3-custom.cfg', help="YOLO config path")
            parser.add_argument("--weights", default='models/DarkNet/fire_detection/yolov3-custom_10000.weights', help="YOLO weights path")
            parser.add_argument("--names", default='models/DarkNet/fire_detection/classes.names', help="class names path")
            args = parser.parse_args()
            # Get names of output layers, output for YOLOv3 is ['yolo_16', 'yolo_23']
            def getOutputsNames(net):
                layersNames = net.getLayerNames()
                return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
            
            
            # Darw a rectangle surrounding the object and its class name 
            def draw_pred(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
            
                label = str(classes[class_id])
            
                color = COLORS[class_id]
            
                cv.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
            
                cv.putText(img, label, (x-10,y-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
            
            
            # Load names classes
            classes = None
            with open(args.names, 'r') as f:
                classes = [line.strip() for line in f.readlines()]
            
            #Generate color for each class randomly
            COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
            
            # Define network from configuration file and load the weights from the given weights file
            net = cv.dnn.readNet(args.weights,args.config)
            uploaded_file = st.file_uploader("Choose a video...", type=["mp4"])
            temporary_location = False
            
            if uploaded_file is not None:
                g = io.BytesIO(uploaded_file.read())  ## BytesIO Object
                temporary_location = "testout_simple.mp4"
            
                with open(temporary_location, 'wb') as out:  ## Open temporary file as bytes
                    out.write(g.read())  ## Read bytes into file
            
                # close file
                out.close()
            def load_video(temporary_location):
                """
                """
                cap = cv.VideoCapture(str(temporary_location))
                return cap
            cap = load_video(temporary_location)
            if st.button('Process Video'):
                if temporary_location:
                    my_bar = st.progress(0)
                    for percent_complete in range(100):
                        time.sleep(0.01)
                        my_bar.progress(percent_complete + 1)
                    image_placeholder = st.empty()
                    while cv.waitKey(1) < 0:
                        try:
                            hasframe, image = cap.read()
                            image=cv.resize(image, (608, 608)) 
                            
                            blob = cv.dnn.blobFromImage(image, 1.0/255.0, (608,608), [0,0,0], True, crop=False)
                            Width = image.shape[1]
                            Height = image.shape[0]
                            net.setInput(blob)
                            
                            outs = net.forward(getOutputsNames(net))
                            
                            class_ids = []
                            confidences = []
                            boxes = []
                            conf_threshold = 0.5
                            nms_threshold = 0.4
                            
                            
                            #print(len(outs))
                            
                            # In case of tiny YOLOv3 we have 2 output(outs) from 2 different scales [3 bounding box per each scale]
                            # For normal normal YOLOv3 we have 3 output(outs) from 3 different scales [3 bounding box per each scale]
                            
                            # For tiny YOLOv3, the first output will be 507x6 = 13x13x18
                            # 18=3*(4+1+1) 4 boundingbox offsets, 1 objectness prediction, and 1 class score.
                            # and the second output will be = 2028x6=26x26x18 (18=3*6) 
                            
                            for out in outs: 
                                #print(out.shape)
                                for detection in out:
                                    
                                #each detection  has the form like this [center_x center_y width height obj_score class_1_score class_2_score ..]
                                    scores = detection[5:]#classes scores starts from index 5
                                    class_id = np.argmax(scores)
                                    confidence = scores[class_id]
                                    if confidence > 0.5:
                                        center_x = int(detection[0] * Width)
                                        center_y = int(detection[1] * Height)
                                        w = int(detection[2] * Width)
                                        h = int(detection[3] * Height)
                                        x = center_x - w / 2
                                        y = center_y - h / 2
                                        class_ids.append(class_id)
                                        confidences.append(float(confidence))
                                        boxes.append([x, y, w, h])
                            
                            # apply  non-maximum suppression algorithm on the bounding boxes
                            indices = cv.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
                            
                            for i in indices:
                                i = i[0]
                                box = boxes[i]
                                x = box[0]
                                y = box[1]
                                w = box[2]
                                h = box[3]
                                draw_pred(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
                           
                            # Put efficiency information.
                            t, _ = net.getPerfProfile()
                            label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
                            cv.putText(image, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
                            image_placeholder.image(image, channels="BGR", use_column_width=True)
                        except:
                            image_placeholder = st.empty()
                            break

            
            
            #--------------------------------------------#
            #--------------------------------------------#

        else:
            raise ValueError("Please select one source from the list")



    def load_image_or_video(self):
        """
        Handle the data input from the user parameters
        """
        if self.guiParam['appType'] == 'Image Applications':
            self.data = self.load_image_source()

        else:
            raise ValueError(
                '[Error] Please select of the two Application pipelines')

        return self.data