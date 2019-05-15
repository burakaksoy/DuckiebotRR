# -*- coding: utf-8 -*-
"""
Created on Wed May  8 17:32:18 2019

@author: BurakAksoy
"""

from RobotRaconteur.Client import *
from mvnc import mvncapi as mvnc

import time
import thread
import numpy as np
import cv2
import sys
import math
from math import floor, atan2, pi, cos, sin, sqrt

#########################################################################
# FUNCTIONS FOR CAMERA RR: BEGIN
#Function to take the data structure returned from the Webcam service
#and convert it to an OpenCV array
def WebcamImageToMat(image):
    frame2=image.data.reshape([image.height, image.width, 3], order='C')
    return frame2

#This function is called when a new pipe packet arrives
current_frame=None
def new_frame(pipe_ep):
    global current_frame
    #Loop to get the newest frame
    while (pipe_ep.Available > 0):
        #Receive the packet
        image=pipe_ep.ReceivePacket()
        #Convert the packet to an image and set the global variable
        current_frame=WebcamImageToMat(image)
# This function connects to camera, starts streaming and updates the current frame in current_frame

def connect_camera(url):
    #Startup, connect, and pull out the camera from the objref
    RRN.UseNumPy=True
    c_host=RRN.ConnectService(url)
    c=c_host.get_Webcams(0)
    #Connect the pipe FrameStream to get the PipeEndpoint p
    p=c.FrameStream.Connect(-1)
    #Set the callback for when a new pipe packet is received to the new_frame function
    p.PacketReceivedEvent+=new_frame
    try:
        c.StartStreaming()
    except: pass
    return p,c
    
def connect_camera2(url):
    #Startup, connect, and pull out the camera from the objref
    RRN.UseNumPy=True
    c_host=RRN.ConnectService(url)
    c=c_host.get_Webcams(0)
    return c

def disconnect_camera(p,c):
    p.Close()
    c.StopStreaming()
# FUNCTIONS FOR CAMERA RR: END
########################################################################

#########################################################################
# FUNCTIONS FOR DRIVE RR: BEGIN
def connect_drive(url):
    #Instruct Robot Raconteur to use NumPy
    RRN.UseNumPy=True
    #Connect to the service
    c=RRN.ConnectService(url)
    return c
# FUNCTIONS FOR DRIVE RR: END
########################################################################

#########################################################################
# FUNCTIONS FOR LANE CONTROLLER: BEGIN
# # Calculate w and v from d and phi
def lane_controller(d,phi):
    #-------------------PARAMETERS: BEGIN
    v_bar = 0.3864
    k_d = -10.30
    k_theta = -5.15
    d_thres =  0.2615
    d_offset = 0.0      
    #-------------------PARAMETERS: END

    # Helper functions: BEGIN--------------------------
    # Helper functions: END--------------------------
    cross_track_err = d - d_offset
    heading_err = phi
    
    v = v_bar/2.0
    
    if math.fabs(cross_track_err) > d_thres:
        cross_track_err = cross_track_err / math.fabs(cross_track_err) * d_thres
    
    omega =  (k_d * cross_track_err + k_theta * heading_err)/1.25
    
    return v,omega
# FUNCTIONS FOR LANE CONTROLLER: END
########################################################################

#########################################################################
# FUNCTIONS FOR INVERSE KINEMATICS: BEGIN
# # Calculate each wheel speed from w and v
def inverse_kinematics(v,omega):
    #-------------------PARAMETERS: BEGIN
    baseline = 0.11
    gain = 1.0
    k = 27.0
    limit = 1.0
    radius = 0.0318
    trim = 0.01   
    #-------------------PARAMETERS: END

    # Helper functions: BEGIN--------------------------
    # Helper functions: END--------------------------
    
    # assuming same motor constants k for both motors
    k_r = k
    k_l = k
    
    # adjusting k by gain and trim
    k_r_inv = (gain + trim) / k_r
    k_l_inv = (gain - trim) / k_l
    
    omega_r = (v + 0.5 * omega * baseline) / radius
    omega_l = (v - 0.5 * omega * baseline) / radius
    
    # conversion from motor rotation rate to duty cycle
    u_r = omega_r * k_r_inv
    u_l = omega_l * k_l_inv

    # limiting output to limit, which is 1.0 for the duckiebot
    vel_right = max(min(u_r, limit), -limit)
    vel_left = max(min(u_l, limit), -limit)
    return vel_right,vel_left
# FUNCTIONS FOR INVERSE KINEMATICS: END
########################################################################

def main():
    url_cam = 'rr+tcp://192.168.43.141:2355?service=Webcam'
    url_drive = 'rr+tcp://192.168.43.141:2356?service=Drive'
    
    # Connect to camera and start streaming on global current_frame variable
    # p,cam = connect_camera(url_cam)
    cam = connect_camera2(url_cam)
    
    # Connect to motors to drive
    car = connect_drive(url_drive)

    ## Check Mividius Device 
    devices = mvnc.EnumerateDevices()
    if len(devices) == 0:
        print('No devices found')
        quit()
    
    device = mvnc.Device(devices[0])
    device.OpenDevice()
    
    # Load Deep Neural Network graph
    with open('./graph', mode='rb') as f:
        graphfile = f.read()
    # Send graph to the device
    graph = device.AllocateGraph(graphfile)
    
    # Image parameters    
    image_size =  np.array([120,160])
    top_cutoff = 40 
    
    is_view = False
    if is_view:
        cv2.namedWindow("Image") # Debug
        
    try:
        prev_time = time.time()
        iteration_times = []
        predict_times = []
        prep_times = []
        
        while True:
            #Just loop resetting the frame
            #This is not ideal but good enough for demonstration.  
            current_frame=WebcamImageToMat(cam.CaptureFrame())          
            if not current_frame is None:          
                # Use image from now on to prevent unknown updates on current frame.
                frame = current_frame
                
                # print("Get-image: "+str(time.time() - prev_time))
                # Resize image
                hei_original, wid_original = frame.shape[0:2]                
                
                if image_size[0] != hei_original or image_size[1] != wid_original:
                    frame = cv2.resize(frame, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)            
                # Crop image
                frame = frame[top_cutoff:,:,:]                
                
                # print("Preprocess0: "+str(time.time() - prev_time))
                
                # Convert color due to openCV defaults BGR to RGB
                img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                
                # print("Preprocess1: "+str(time.time() - prev_time))
                # Convert image to float16 and normalize so that device can process
                img = img.astype(np.float16)/255.0
                
                #print("Preprocess: "+str(time.time() - prev_time))
                prep_times.append(time.time() - prev_time) 
                
                # Send image to the device, and Calculate d and phi
                graph.LoadTensor(img, 'user object')
                output, userobj = graph.GetResult()                
                output = output.astype(np.float32)
                # Extract d and phi              
                d = output[0]
                phi= output[1]
                print('Output(d,phi): ' + str(output))

                # If d or phi could not be find, stop the car.
                if d is None or phi is None:
                    # car.setWheelsSpeed(0,0)
                    prev_time = time.time()
                    continue
                                               
                # Calculate w(omega) and v from d and phi
                v,w = lane_controller(d,phi)
                #print(v,w)
                
                # Calculate inverse kinematics to find each wheel speed
                vel_right,vel_left = inverse_kinematics(v,w)
                # print("vel_left: " + str(vel_left) + ", vel_right: " + str(vel_right))
                
                # print("Predict: "+str(time.time() - prev_time))
                predict_times.append(time.time() - prev_time) 
                
                # Drive the car
                car.setWheelsSpeed(vel_left,vel_right)
                # car.setWheelsSpeed(0,0)
                
                # Calculate passed time rate
                duration_rate = (time.time() - prev_time)
                iteration_times.append(duration_rate)
                prev_time = time.time()
                
                # Print the fps
                print("Rate: "+str(1.0/duration_rate))
                
                # View for Debug
                if is_view:
                    frame = cv2.resize(frame, (640, 320))
                    cv2.imshow("Image-with-lines",frame)
                    if cv2.waitKey(1)!=-1:
                        break
    except KeyboardInterrupt:
        print('Interrupted!')
        
    # Convert fps values to numpy array
    iteration_times = np.array(iteration_times)
    avr_times = np.mean(iteration_times)
    
    prep_times = np.array(prep_times)
    avr_prep_times = np.mean(prep_times)
    
    predict_times = np.array(predict_times)
    avr_predict_times = np.mean(predict_times)
    
    # Print average fps
    print("Avr. (get image + preprocess) time: " + str(avr_prep_times*1000) + ' ms')
    print("Avr. (prediction + motor speed control) time: " + str((avr_predict_times-avr_prep_times)*1000) + ' ms')
    print("Avr. (set wheels speed) time: " + str((avr_times-avr_predict_times)*1000) + ' ms')
    print("Avr. loop rate: " + str(1.0/avr_times) + ' FPS')

    print('Shutting Down..')
    car.setWheelsSpeed(0,0)
    # disconnect_camera(p,cam)
    
    # Clear and Disconnect Movidius device
    graph.DeallocateGraph()
    device.CloseDevice()
    print('Finished')

if __name__ == '__main__':
    main()
