
from cgitb import enable
from typing import final
from charset_normalizer import detect
import cv2 as cv
import numpy as np
import time
import imutils
import os
import discord
import multiprocessing
import pynput
from pynput.keyboard import Key, Controller


class ERC:

    def __init__(self, video, config, model, classes):
        self.videoPath = video
        self.configPath = config
        self.modelPath = model
        self.classesPath = classes


        self.net = cv.dnn_DetectionModel(self.modelPath, self.configPath)
        self.net.setInputSize(320,320)
        self.net.setInputScale(1.0/127.5)
        self.net.setInputMean((127.5,127.5,127.5))
        self.net.setInputSwapRB(True)

        self.readClasses()  
    

    def readClasses(self):
            with open(self.classesPath, 'r') as f:
                self.classesList = f.read().splitlines()

            self.classesList.insert(0, '__Background__')



    def onVideo(self):
       

        detectionEnabled = False
        capture = cv.VideoCapture(self.videoPath)
        glitchOn = False
        

        # if (capture.isOpened()==False):
        #     print("huge L fr fr")
        #     return
        
        (succ, image) = capture.read()
        
 

        startTime = 0

        while succ:
            
            currentTime = time.time()
            fps = 1/(currentTime-startTime)
            startTime = currentTime



            
  
            mask = np.zeros((image.shape[0],image.shape[1],1), dtype = 'uint8')
            IDs, confidences, bboxs = self.net.detect(image, confThreshold = 0.3)

            bboxs = list(bboxs)
            confidences = list(np.array(confidences).reshape(1,-1)[0])
            confidences = list(map(float,confidences))

            bboxIDx = cv.dnn.NMSBoxes (bboxs, confidences, score_threshold = 0.5, nms_threshold = 0.2)
            #print (str(detectionEnabled)+ "1")
            if (len(bboxIDx) != 0) & (detectionEnabled == True):
                for i in range (0, len(bboxIDx)):
                    
                    bbox = bboxs[np.squeeze(bboxIDx[i])]
                    classConfidence = confidences[np.squeeze(bboxIDx[i])]
                    CL = np.squeeze(IDs[np.squeeze(bboxIDx[i])])
                    classLabel = self.classesList[CL]

                    x,y,w,h = bbox
                    
                    enlargeW = 1+w*1.2
                    enlargeH = 1+h*1.2
                    

                    disWidth = w
                    disHeight = h

                    #cv.putText(image, str(int(disWidth)), (20, 200), cv.FONT_HERSHEY_COMPLEX, 2, (0,255,0), 2 )
                    # Determining if Height or width is a better indicator of distance for myself
                    # print("Width:" + str(disWidth))
                    # print("Height:" + str(disHeight))
                    # print("Area:" + str(int(disWidth)*int(disHeight)))


                    if disWidth > 120: #
                        glitchOn = True
                        
                    else:
                        glitchOn = False
                    

                    if classLabel == "person":
                        
                        cv.rectangle(mask, (x-20,y-30), (int(x+enlargeW), int(y+enlargeH)), color= (255,255,255), thickness=-1)


                        

            
            
            output2 = cv.inpaint(image, mask, 9, cv.INPAINT_NS)
            #blur = cv.GaussianBlur(output2, (5,5), cv.BORDER_DEFAULT)
            blur = cv.medianBlur(output2, 3)
            #print (str(detectionEnabled) + "2")
            
            finalOut = blur
            
            if glitchOn == True:

                # Glitch effects use makeworld-the-better-one's live-glitch
                # https://github.com/makeworld-the-better-one/live-glitch
                width_pc=1/100
                darken=15

                h, w = blur.shape[:2]
                # Number of scanlines
                n_lines = int(h / (width_pc * 100))
                width_px = int(h * width_pc)

                for i in range(n_lines):
                    dark_line = np.maximum(blur[i*width_px*2 : i*width_px*2 + width_px, :].astype("int16") - darken, 0).astype("uint8")
                    blur[i*width_px*2 : i*width_px*2 + width_px, :] = dark_line


                variance = 10

                row, col, ch = blur.shape
                mean = 0
                sigma = variance ** 1
                gauss = np.random.normal(mean, sigma, (row, col, ch))
                gauss = gauss.reshape(row, col, ch)
                noisy = blur + gauss
                cv.normalize(noisy, noisy, 0, 255, cv.NORM_MINMAX)

                noise = noisy.astype("uint8")
                


                spacing = 15

                gray = cv.cvtColor(noise, cv.COLOR_BGR2GRAY)
                sat = np.full(gray.shape, 100, dtype="uint8")
                blue_hue = np.full(gray.shape, 180//2, dtype="uint8")  
                red_hue = np.full(gray.shape, 0, dtype="uint8")
                blue = cv.merge(np.array([blue_hue, sat, gray], dtype="uint8"))
                red = cv.merge(np.array([red_hue, sat, gray], dtype="uint8"))
                blue = cv.cvtColor(blue, cv.COLOR_HSV2BGR)
                red = cv.cvtColor(red, cv.COLOR_HSV2BGR)

                blue = imutils.translate(blue, spacing, 0)  
                red = imutils.translate(red, -spacing, 0) 
                np.copyto(noise, blue, where=(blue > noise))
                np.copyto(noise, red, where=(red > noise))
                result_hsv = cv.cvtColor(noise, cv.COLOR_BGR2HSV).astype("float32")
                h, s, v = cv.split(result_hsv)
                s *= 1.5 
                s = np.clip(s, 0, 255)
                v += 20 
                v = np.clip(v, 0, 255)
                result_hsv = cv.merge([h, s, v])
                finalOut = cv.cvtColor(result_hsv.astype("uint8"), cv.COLOR_HSV2BGR)
    
            
                    
            cv.putText(image, "FPS: "+ str(int(fps)), (20, 70), cv.FONT_HERSHEY_COMPLEX, 2, (0,255,0), 2 )
            

            cv.imshow('normalTrack',image)
            cv.imshow('mask', mask)
            
            #cv.imshow('TELEA',output1)
            cv.imshow('NS', finalOut)

            (succ, image) = capture.read()

            key = cv.waitKey(1) & 0xFF

            if (key == ord("q")):
                detecting.terminate()   
        

            if (key == ord("d")):
                if (detectionEnabled == True):
                    detectionEnabled = False
                    glitchOn = False
                else:
                    detectionEnabled = True


            
        cv.destroyAllWindows()



def vigil():

    videoPath = 0 #webcam

    configPath = os.path.join("model_data", r"D:\VSC Projects\Vigil\model\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    modelPath = os.path.join("model_data", r"D:\VSC Projects\Vigil\model\frozen_inference_graph.pb")
    classesPath = os.path.join("model_data", r"D:\VSC Projects\Vigil\model\coco.names")

    detection = ERC (videoPath, configPath, modelPath, classesPath)
    detection.onVideo()


def disc():
    keyboard = Controller()

    TOKEN = 'MTAzMzkxMzQzMjIwMzg1Mzg2NA.GjKmk0.cq016otiOpx5LRGyotAe52VHcMDmNGfPVEbOEE'
    intents = discord.Intents.all()
    client = discord.Client(intents=intents)

    class discordBotERC7:
       

        @client.event
        async def on_ready():
            print ('We have logged in as {0.user}, ERC-7'.format(client))

        @client.event
        async def on_message(message):
            username = str(message.author).split('#')[0]
            user_message = str(message.content)
            channel = str(message.channel.name)
            
            print(f'{username}: {user_message} ({channel})')

            if message.author == client.user:
                return

            if message.channel.name == 'vigil-bot':
                if user_message.lower() == 'close camera':
                    await message.channel.send(f'Okay {username}, closing the camera')
                    keyboard.press('q')
                    keyboard.release('q')
                    return

                if user_message.lower() == 'open camera':
                    await message.channel.send(f'Okay {username}, opening the camera')
                    detecting.start()

                    return

                if user_message.lower() == 'toggle detection':
                    await message.channel.send(f'Okay {username}, toggling ERC-7!')
                    keyboard.press('d')
                    keyboard.release('d')

                    return
                

    client.run(TOKEN)

detecting = multiprocessing.Process(target=vigil)
discordBotProcess = multiprocessing.Process(target=disc)

if __name__ == '__main__':
    detecting.start()
    discordBotProcess.start()

        

