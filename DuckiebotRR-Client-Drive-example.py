#Example iRobot Create client in Python

from RobotRaconteur.Client import *
import time
import numpy
import sys


def main():
    url='rr+tcp://duckiepark:2356?service=Drive'
    if (len(sys.argv)>=2):
        url=sys.argv[1]

    #Instruct Robot Raconteur to use NumPy
    RRN.UseNumPy=True

    #Connect to the service
    c=RRN.ConnectService(url)

    #Drive a bit
    c.setWheelsSpeed(0.5,0.5)
    time.sleep(5)
    c.setWheelsSpeed(0,0)
    time.sleep(10)

if __name__ == '__main__':
    main()
