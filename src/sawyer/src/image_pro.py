#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from lab4_cam.srv import ImageSrv, ImageSrvResponse
import cv2, time, sys
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from numpy.linalg import *
import math
import sys
import copy
from ar_track_alvar_msgs.msg import AlvarMarker, AlvarMarkers
from tf2_msgs.msg import TFMessage
import message_filters


# Nominal length of a tile side
DRAW_LENGTH = 100 #cm
DRAW_WIDTH = 150
bridge = CvBridge()
approx=0
nx = 1
ny = 1

# Helper function to check computed homography
# This will draw dots in a grid by projecting x,y coordinates
# of tile corners to u,v image coordinates
def check_homography(image, H, nx, ny, length=DRAW_LENGTH):
  # H should be a 3x3 numpy.array
  # nx is the number of tiles in the x direction
  # ny is the number of tiles in the y direction
  # length is the length of one side of a tile
  # image is an image array
  for i in range(nx+1):
    for j in range(ny+1):
      xbar = np.array([[i*length],[j*length],[1]])
      ubar = np.dot(H,xbar).T[0]
      u = np.int(ubar[0]/ubar[2])
      v = np.int(ubar[1]/ubar[2])
      print 'Dot location: ' + str((u,v))
      cv2.circle(image, (u,v), 5, 0, -1)
    cv2.imshow('Check Homography', image)

class ar_tracking:

    def __init__(self):
        self.total_markers = 1
        self.subscriber1 = None
        self.subscriber2 = None
        self.markers = None
        self.image_sub = None
        self.bridge = CvBridge()
        self.save=None
        self.image=None
    def function(self,data):
      print "xxxxxxxxxxxxxxxxxxxxxxxxxxx"
      try:
        img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        img2 = img
      except CvBridgeError as e:
        print(e)
      global approx
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
      cv2.imshow('thresh', thresh)
      dst = cv2.cornerHarris(thresh, 2, 3, 0.04)
      dst = cv2.dilate(dst,None)
      cv2.imshow('dst', img)
      self.image = img

      contours,h = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
      rect = None
      for cnt in contours:
        approx = cv2.approxPolyDP(cnt,0.02*cv2.arcLength(cnt,True),True)
        if len(approx)==4:
          if cv2.contourArea(approx) > 700 and cv2.contourArea(approx) < 10000:
            self.save = approx
            #cv2.drawContours(img2,[approx],0,(0,255,0),-1)
      print "cccccccccccccccccccccccccc"
      cv2.imshow('img2', img2)
      cv2.waitKey(100)
      rospy.signal_shutdown("BYE")

      # cv2.destroyAllWindows()
    def callback(self, message):

        if len(message.markers) >= self.total_markers:
            self.markers = message.markers
            #self.subscriber1.unregister()
            #rospy.signal_shutdown("ar tags found")

    def begin_tracking(self):
        #self.image_sub=rospy.Subscriber("io/internal_camera/head_camera/image_raw",Image,self.function,queue_size=1000)
        #self.subscriber1 = rospy.Subscriber("/ar_pose_marker", AlvarMarkers, self.callback,queue_size=10)
        sub1 = message_filters.Subscriber("/ar_pose_marker", AlvarMarkers)
        sub1.registerCallback(self.callback)

        sub2 = message_filters.Subscriber("io/internal_camera/head_camera/image_raw", Image)
        sub2.registerCallback(self.function)

        rospy.spin()
        return self.markers,self.save, self.image
def ros_to_np_img(ros_img_msg):
  return np.array(bridge.imgmsg_to_cv2(ros_img_msg,'bgr8'))

# Define the total number of clicks we are expecting (4 corners)
TOT_CLICKS = 1

if __name__ == '__main__':
  
  # Waits for the image service to become available
  #rospy.wait_for_service('last_image')
  # Initializes the image processing node
  rospy.init_node('image_processing_node')  
  #last_image_service = rospy.ServiceProxy('last_image', ImageSrv)
  tracker = ar_tracking()

  markers, contours, img = tracker.begin_tracking()
  cv2.drawContours(img,[contours],0,(0,255,0),-1)
  print (contours)

  contours = np.vstack(contours).squeeze()
  #try:
  # rospy.spin()
  #except KeyboardInterrupt:
  # print("shuutting down")
  #def on_mouse_click(event,x,y,flag,param):
  #if(event == cv2.EVENT_LBUTTONUP):
    #point = (x,y)
    #print "Point Captured: " + str(point)
    #points.append(point)
  #while not rospy.is_shutdown():
  np_image = img
  cv2.imshow("CV Image", np_image)
  #cv2.setMouseCallback("CV Image", on_mouse_click, param=1)
  points=contours

  cv2.waitKey(1000)
  uv = np.array(contours).T
  #xy = np.array(contours).T
  print (uv)
  Des_x = (uv[0][0]+uv[0][2])/2
  Des_y = (uv[1][0]+uv[1][2])/2
  print(Des_x,Des_y)
  u=uv[0]
  v=uv[1]
  H = np.eye(3)
  [x1,y1]=[0,0]
  [x2,y2]=[nx*DRAW_WIDTH,0]
  [x3,y3]=[nx*DRAW_WIDTH,ny*DRAW_LENGTH]
  [x4,y4]=[0,ny*DRAW_LENGTH]
  A=np.array(
  [[x1,y1,1.,0.,0.,0.,-u[0]*x1,-u[0]*y1],
  [0,0,0,x1,y1,1,-v[0]*x1,-v[0]*y1],
  [x2,y2,1,0,0,0,-u[1]*x2,-u[1]*y2],
  [0,0,0,x2,y2,1,-v[1]*x2,-v[1]*y2],
  [x3,y3,1,0,0,0,-u[2]*x3,-u[2]*y3],
  [0,0,0,x3,y3,1,-v[2]*x3,-v[2]*y3],
  [x4,y4,1,0,0,0,-u[3]*x4,-u[3]*y4],
  [0,0,0,x4,y4,1,-v[3]*x4,-v[3]*y4]])
  B=np.reshape(np.array([u[0],v[0],u[1],v[1],u[2],v[2],u[3],v[3]]),(8,1))
  H_n=np.reshape(np.dot(np.linalg.inv(A),B),(8,1))
  one=np.array([[1]])
  H_w=np.concatenate((H_n,one))
  H_q=np.reshape(H_w,(9,1))
  H=np.reshape(np.array([H_q]), (3, 3))
  Q=np.linalg.inv(H)
  Des=np.array([Des_x,Des_y,1])
  xpo=np.dot(Q[0],Des)/np.dot(Q[2],Des)
  ypo=np.dot(Q[1],Des)/np.dot(Q[2],Des)
  print xpo,ypo
  key = -1
  check_homography(img, H, nx, ny)