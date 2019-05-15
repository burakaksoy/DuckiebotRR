#Simple example Robot Raconteur webcam client
#This program will capture a frame from both webcams and show it
#on the screen
from RobotRaconteur.Client import *
import time
import thread
import numpy
import cv2
import sys

#Function to take the data structure returned from the Webcam service
#and convert it to an OpenCV array
def WebcamImageToMat(image):
    frame2=image.data.reshape([image.height, image.width, 3], order='C')
    return frame2

#Main program
def main():

    url='rr+tcp://192.168.43.141:2355/?service=Webcam'
    is_view = True
    if (len(sys.argv)>=2):
        url=sys.argv[1]

    #Start up Robot Raconteur and connect, standard by this point
    RRN.UseNumPy=True
    c_host=RRN.ConnectService(url)

    #Use objref's to get the cameras. c_host is a "WebcamHost" type
    #and is used to find the webcams
    c1=c_host.get_Webcams(0)
    if (is_view):
        cv2.namedWindow("Image")

    finish_time = 0
    while True:
        #Pull a frame from each camera, c1 and c2
        frame1=WebcamImageToMat(c1.CaptureFrame())
        
        if (not frame1 is None):
            start_time = time.time()
            if (is_view):
                #Show the images
                cv2.imshow("Image",frame1)
            # Print the fps
            print(str(1.0/(start_time - finish_time)))
            finish_time = time.time()
        if cv2.waitKey(1)!=-1:
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
