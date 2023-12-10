#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#######


#  i cant decide if the package should be a ros1 or ros2 package


######



"""
This demo calculates multiple things for different scenarios.
Here are the defined reference frames:
TAG:
                A y
                |
                |
                |tag center
                O---------> x
CAMERA:
                X--------> x
                | frame center
                |
                |
                V y
F1: Flipped (180 deg) tag frame around x axis
F2: Flipped (180 deg) camera frame around x axis
The attitude of a generic frame 2 respect to a frame 1 can obtained by calculating euler(R_21.T)
We are going to obtain the following quantities:
    > from aruco library we obtain tvec and Rct, position of the tag in camera frame and attitude of the tag
    > position of the Camera in Tag axis: -R_ct.T*tvec
    > Transformation of the camera, respect to f1 (the tag flipped frame): R_cf1 = R_ct*R_tf1 = R_cf*R_f
    > Transformation of the tag, respect to f2 (the camera flipped frame): R_tf2 = Rtc*R_cf2 = R_tc*R_f
    > R_tf1 = R_cf2 an symmetric = R_f


export ROS_DOMAIN_ID=2
hazeezadebayo@hazeezadebayo:~$ source /opt/ros/humble/setup.bash; source /home/hazeezadebayo/colcon_ws/install/setup.bash; ros2 run nav2_rosdevday_2021 aruco_publisher.py
"""



from re import L
import os, warnings, yaml, time, sys, ast, math, signal, psycopg2, psycopg2.extras, collections, cv2 # , argparse, threading, csv,
from rclpy.node import Node
from datetime import datetime
from geometry_msgs.msg import  PoseStamped # Pose
from tf2_ros import TransformBroadcaster # , StaticTransformBroadcaster
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
#from cv_bridge import CvBridge
from cv2 import aruco
import numpy as np
from math import atan2, sin, cos, radians, acos, degrees
from scipy.spatial.transform import Rotation as Ro
from std_msgs.msg import String, Float32MultiArray
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
import pyrealsense2 as rs
 

import rospy, rclpy  # -----> ros1 ros2?
# from mono_vo.srv import Monovo, MonovoResponse
from mono_vo.mono_vo_ import MonoVisualOdometry


class Aruco_pub(Node):
    def __init__(self):
        super().__init__('Aruco_pub')

        device_serial_no = '843112072968'
        self.mono_vo_initialized = False
        

        # --- 180 deg rotation matrix around the x axis
        self._R_flip      = np.zeros((3,3), dtype=np.float32)
        self._R_flip[0,0] = 1.0
        self._R_flip[1,1] =-1.0
        self._R_flip[2,2] =-1.0

        # ar_tags = ast.literal_eval(str(agv_dict["ar_tags"]))
        # self.odom_frame = rospy.get_param('~odom_frame', 'odom')

        self.ar_tags = eval(rospy.get_param('~ar_tags_info', ['x','y','z']))
        self.ar_tags = [float(self.ar_tags[0]), float(self.ar_tags[1]), float(self.ar_tags[2])] 

        aruco_id, aruco_x, aruco_y, aruco_th = float(0), float(0), float(0), float(0), float(0) 

        self.aruco_stat_time = time.time_ns()
        self.msg_as = Float32MultiArray()
        self.aruco_stat_pub = self.create_publisher(Float32MultiArray, 'aruco_stat', 1)

        # ---------------------------------------

        marker_size = 13.5 # [-cm] 
        self.current_detected_id = 'none'
        
        
        self.ids_to_find = {"ids":"x_bw, y_bw, t_bw",
                            "1":[9.0, 0.0, 180.0],  # -  tested: directly opposite agv  
                            "2":[9.0, -5.0, 135.0], # \  tested:
                            "3":[10.5, -4.6, 180.0],  # /  tested: slanted at 45 degrees in agv view direction
                            "4":[1.0, 1.5, -90.0],  # |  tested: 90 degrees to agv view      
                            "5":[]}
        # self.ids_to_find = {}
        # if (len(ar_tags) != 0):
        #     for tag in ar_tags:
        #         self.ids_to_find[str(int(tag[0]))] = [float(tag[1]), float(tag[2]), float(tag[3])]
        print('ids_to_find: ', self.ids_to_find)


        if device_serial_no != 'none':
            ctx = rs.context()
            if len(ctx.devices) > 0:
                for d in ctx.devices:
                    print ('Found device: ', \
                            d.get_info(rs.camera_info.name), ' ', \
                            d.get_info(rs.camera_info.serial_number))

                    if device_serial_no != d.get_info(rs.camera_info.serial_number):

                        show_video = True

                        # define pipeline
                        pipeline = rs.pipeline()
                        config = rs.config()

                        config.enable_device(d.get_info(rs.camera_info.serial_number)) 

                        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
                        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
                        
                        # Start streaming
                        pipeline.start(config)

                        self.rs_aruco_pose(pipeline, marker_size, show_video)

            else:
                print("No Intel Device connected")
        else:
            self.generic_aruco_pose(pipeline, marker_size, show_video)


    # -----------------------------------------
    # -----------------------------------------
    # -----------------------------------------

    def pub_aruco_info(self):
        if ((time.time_ns() - self.aruco_stat_time)*1e-9) > 80: # 60s=1min
            self.aruco_stat_time = time.time_ns()
            self.msg_as.data = [aruco_id, aruco_x, aruco_y, aruco_th]
            self.aruco_stat_pub.publish(self.msg_as)
            # self.get_logger().info('Publishing aruco stat:  "%f" "%f" "%f" "%f" "%f"' % (self.msg_as.data[0], self.msg_as.data[1], self.msg_as.data[2], self.msg_as.data[3], self.msg_as.data[4]))

    # -----------------------------------------
    # -----------------------------------------
    # -----------------------------------------

    def _rotationMatrixToEulerAngles(self, R):
    # Calculates rotation matrix to euler angles
    # The result is the same as MATLAB except the order
    # of the euler angles ( x and z are swapped ).
    
        def isRotationMatrix(R):
            Rt = np.transpose(R)
            shouldBeIdentity = np.dot(Rt, R)
            I = np.identity(3, dtype=R.dtype)
            n = np.linalg.norm(I - shouldBeIdentity)
            return n < 1e-6        
        assert (isRotationMatrix(R))

        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0

        return np.array([x, y, z])

    # -----------------------------------------
    # -----------------------------------------
    # -----------------------------------------

    def euler_to_quaternion(self, yaw, pitch, roll):
        qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
        qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
        qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        return [qx, qy, qz, qw]


    def generic_aruco_pose(self, marker_size, show_video=None):

        marker_dict = aruco.Dictionary_get(aruco.DICT_5X5_250)
        param_markers = aruco.DetectorParameters_create()

        # Load images
        img1 = cv2.imread('image1.jpg',0)  # queryImage
        img2 = cv2.imread('image2.jpg',0)  # trainImage

        # Get image dimensions
        height, width = img1.shape

        # Estimate principal point (assume it's at the center of the images)
        ppx = width / 2
        ppy = height / 2

        # Estimate focal length (assume a certain field of view)
        fov = 60  # field of view in degrees, adjust based on your camera specifications
        # fx = img1.shape[1] / (2 * np.tan(np.radians(fov / 2)))
        # fy = fx  # assume square pixels
        # If the image is not square, the focal length in the x and y directions might be different.
        # Here we assume that the field of view is specified for the diagonal of the image.
        diagonal = np.sqrt(width**2 + height**2)
        fx = fy = diagonal / (2 * np.tan(np.radians(fov / 2)))

        # Print estimated camera matrix
        print('Estimated camera matrix:')
        print(f'fx: {fx}, fy: {fy}, ppx: {ppx}, ppy: {ppy}')

        # print(np.array([[fx, 0, ppx], [0, fy, ppy], [0, 0, 1]]))
        cam_mat = np.array([[fx, 0, ppx],
                            [0, fy, ppy],
                            [0, 0, 1]])
        
        # Define the distortion coefficients d
        distCoef = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        # you need to make sure this while loop only runs after the
        # second image is obtained not with the first as the cam_matrix
        # might have not been estimated.
        while True:
            
            #-- Read the camera frame
            ret, frame = self._cap.read()
            
            #-- Convert in gray scale
            gray_image    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #-- remember, OpenCV stores color images in Blue, Green, Red

            #                       VISUAL ODOMETRY
            #########################################################
            #-------------------------------------------------------#
            #########################################################
            # Initialize the ICP process object
            # ppx = ppx
            # ppy = ppy
            # fx = fx 
            # fy = fy
            frame_threshold = 10
            keypoint_threshold = 500 # a parameter that determines when to trigger a new feature detection
            cam_to_ground_height = 0.5 # [m] height of robot camera from ground
            detector_type = 'SIFT'
            matcher_type = 'BF'
            
            if self.mono_vo_initialized == False:
                self.mono_vo_initialized = True
                mono_vo = MonoVisualOdometry(ppx, ppy, fx, fy, cam_to_ground_height, frame_threshold, keypoint_threshold, detector_type, matcher_type)

            vo_R, vo_t = mono_vo.process_frame() # R = np.eye(3) | # t = np.zeros((3, 1))
            # _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(np.hstack((vo_R, vo_t)))
            # vo_roll, vo_pitch, vo_yaw = [math.radians(_) for _ in euler_angles]
            vo_roll, vo_pitch, vo_yaw = self._rotationMatrixToEulerAngles(vo_R)
            vo_cam = -vo_R.T.dot(vo_t)
            print("Camera Coordinate: roll=%.3lf, pitch=%.3lf, yaw=%.3lf\n" % (vo_roll, vo_pitch, vo_yaw))  
            print("Camera Coordinate: x=%.3lf, y=%.3lf, z=%.3lf\n" % (vo_cam[0], vo_cam[1], vo_cam[2]))  
            #########################################################
            #-------------------------------------------------------#
            #########################################################
            
            #-- Find all the aruco markers in the image
            # marker_corners, marker_IDs, reject = aruco.detectMarkers(gray_image, marker_dict, parameters=param_markers )
            marker_corners, marker_IDs, reject = aruco.detectMarkers(
                            image= gray_image, 
                            dictionary= marker_dict, 
                            parameters= param_markers,
                            cameraMatrix= cam_mat, 
                            distCoeff= distCoef)

            if marker_corners:
                #camera to marker rotation and translation.
                #-- ret = [rvec, tvec, ?]
                #-- array of rotation and position of each marker in camera frame
                #-- rvec = [[rvec_1], [rvec_2], ...]    attitude of the marker respect to camera frame
                #-- tvec = [[tvec_1], [tvec_2], ...]    position of the marker in camera frame
                ret = aruco.estimatePoseSingleMarkers(marker_corners, marker_size, cam_mat, distCoef)
                
                #-- Unpack the output, get only the first
                rvec, tvec = ret[0][0,0,:], ret[1][0,0,:]

                #-- Draw the detected marker and put a reference frame over it
                aruco.drawDetectedMarkers(frame, marker_corners)
                aruco.drawAxis(frame, cam_mat, distCoef, rvec, tvec, 10)

                #-- Print the tag position in camera frame
                str_position = "MARKER Position x=%4.0f  y=%4.0f  z=%4.0f"%(tvec[0], tvec[1], tvec[2])
                cv2.putText(frame, str_position, (0, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2, cv2.LINE_AA)  

                #-- Obtain the rotation matrix tag->camera
                R_ct = np.matrix(cv2.Rodrigues(rvec)[0])  # cv2.Rodrigues(rvec)
                R_tc = R_ct.T

                #-- Get the attitude in terms of euler 321 (Needs to be flipped first)
                roll_marker, pitch_marker, yaw_marker = self._rotationMatrixToEulerAngles(R_tc) # self._R_flip*R_tc
                # quaternion = self.euler_to_quaternion(yaw_marker, pitch_marker, roll_marker)
                str_attitude = "MARKER Attitude r=%4.0f  p=%4.0f  y=%4.0f"%(math.degrees(roll_marker),math.degrees(pitch_marker), math.degrees(yaw_marker))
                cv2.putText(frame, str_attitude, (0, 150), cv2.FONT_HERSHEY_COMPLEX_SMALL , 1, (0, 255, 0), 2, cv2.LINE_AA)

                #-- Now get Position and attitude of the camera respect to the marker
                v_cam = -R_tc*np.matrix(tvec).T # rotate and translate the position of the marker (given by tvec) from the camera frame to the marker frame
                print("Camera Coordinate System: x=%.3lf, y=%.3lf, z=%.3lf\n" % (v_cam[2], v_cam[0], v_cam[1]))  
                        
                total_markers = range(0, marker_IDs.size)
                for ids, corners, i in zip(marker_IDs, marker_corners, total_markers):

                    corners = corners.reshape(4, 2)
                    corners = corners.astype(int)

                    M = cv2.moments(corners)
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])                
                    area = round(int(cv2.arcLength(corners,True))/marker_size, 1)
                    # print('cx, cy: ', cx, cy, area )
                                
                    global aruco_id, aruco_x, aruco_y, aruco_th 
                    aruco_id = float(ids[0])
                    aruco_x, aruco_y, aruco_th = self.estimate_pose(ids, yaw_marker, v_cam)
                            
                    if aruco_x != None and aruco_y != None and aruco_th != None:
                        # define region of interest
                        if (210 < cx < 550) and (3.7 < area < 75.8): 
                            print("id: "+str(ids[0])+": measured cx "+str(cx)+", cy "+str(cy)+" and area "+str(area)+", within good visible range. node will publish robot's estimated location.") 
                            self.pub_aruco_info()
                        else:
                            pass
                            
                    cv2.putText(frame, f"Area:{area}", (cx-120,cy-80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,255,255), 1)
                    # cv2.putText(frame,f"{depth_in_meters}",(cx,cy),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,150,255),2)
                    cv2.putText(frame, f"id: {ids[0]}", (30,30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 2)
                    # cv2.circle(color_image,(cx,cy),2,(0,0,255),-1)       
    
            if show_video:
                #--- Display the frame: Show images 
                cv2.namedWindow('camera-RGB', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('frame', frame)

                #--- use 'q' to quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self._cap.release()
                    cv2.destroyAllWindows()
                    break



    # -----------------------------------------
    # -----------------------------------------
    # -----------------------------------------

    def rs_aruco_pose(self, pipeline, marker_size, show_video):
        
        marker_dict = aruco.Dictionary_get(aruco.DICT_5X5_250)
        param_markers = aruco.DetectorParameters_create()
                
        try:
            while True:

                frame = pipeline.wait_for_frames()
                depth_frame = frame.get_depth_frame()     
                color_frame = frame.get_color_frame()

                if not depth_frame or not color_frame:
                    sys.stderr.write("Failed to create color frame or depth frame. Quitting.\n")
                    break

                # depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                align = rs.align(rs.stream.color)
                aligned_frame = align.process(frame)

                # Intrinsics & Extrinsics
                # intrinsic of the aligned depth frame (should be equal to the color intrinsic matrix)
                aligned_depth_frame = aligned_frame.get_depth_frame()
                aligned_depth_intrinsic = aligned_depth_frame.profile.as_video_stream_profile().intrinsics 
                # intrinsic = color_frame.profile.as_video_stream_profile().intrinsics
                # print(depth_intrinsic.width, depth_intrinsic.height, depth_intrinsic.ppx 'cx', depth_intrinsic.ppy 'cy', depth_intrinsic.fx, depth_intrinsic.fy, depth_intrinsic.model, depth_intrinsic.coeffs)
                # color_to_depth_extrinsic = color_frame.profile.get_extrinsics_to(depth_frame.profile) #src.get_extrinsics_to(dst)
                # color_to_depth_R =  np.reshape(color_to_depth_extrinsic.rotation, [3,3]).T
                # color_to_depth_T = np.array(color_to_depth_extrinsic.translation)
                # print(color_to_depth_R, color_to_depth_T) 


                #                       VISUAL ODOMETRY
                #########################################################
                #-------------------------------------------------------#
                #########################################################
                # Initialize the ICP process object
                ppx = aligned_depth_intrinsic.ppx
                ppy = aligned_depth_intrinsic.ppy
                fx = aligned_depth_intrinsic.fx 
                fy = aligned_depth_intrinsic.fy
                frame_threshold = 10
                keypoint_threshold = 500 # a parameter that determines when to trigger a new feature detection
                cam_to_ground_height = 0.5 # [m] height of robot camera from ground
                detector_type = 'SIFT'
                matcher_type = 'BF'
                
                if self.mono_vo_initialized == False:
                    self.mono_vo_initialized = True
                    mono_vo = MonoVisualOdometry(ppx, ppy, fx, fy, cam_to_ground_height, frame_threshold, keypoint_threshold, detector_type, matcher_type)

                vo_R, vo_t = mono_vo.process_frame() # R = np.eye(3) | # t = np.zeros((3, 1))
                # _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(np.hstack((vo_R, vo_t)))
                # vo_roll, vo_pitch, vo_yaw = [math.radians(_) for _ in euler_angles]
                vo_roll, vo_pitch, vo_yaw = self._rotationMatrixToEulerAngles(vo_R)
                vo_cam = -vo_R.T.dot(vo_t)
                print("Camera Coordinate: roll=%.3lf, pitch=%.3lf, yaw=%.3lf\n" % (vo_roll, vo_pitch, vo_yaw))  
                print("Camera Coordinate: x=%.3lf, y=%.3lf, z=%.3lf\n" % (vo_cam[0], vo_cam[1], vo_cam[2]))  
                #########################################################
                #-------------------------------------------------------#
                #########################################################

                cam_mat = np.array([[fx, 0, ppx],
                                    [0, fy, ppy],
                                    [0, 0, 1]])

                distCoef = np.array([aligned_depth_intrinsic.coeffs])
                
                gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                # publish on gray_image topic here

                cv2.line(color_image, (280,0),(280,480),(255,255,0),1)
                cv2.line(color_image, (400,0),(400,480),(255,255,0),1)

                #                       ARUCO LOCALIZER
                #########################################################
                #-------------------------------------------------------#
                #########################################################
                marker_corners, marker_IDs, reject = aruco.detectMarkers(gray_image, marker_dict, parameters=param_markers )

                if marker_corners:
                    #camera to marker rotation and translation.
                    ret = aruco.estimatePoseSingleMarkers(marker_corners, marker_size, cam_mat, distCoef)
                    rvec, tvec = ret[0][0,0,:], ret[1][0,0,:]

                    #-- Draw the detected marker and put a reference frame over it
                    aruco.drawDetectedMarkers(color_image, marker_corners)
                    aruco.drawAxis(color_image, cam_mat, distCoef, rvec, tvec, 10)

                    #-- Obtain the rotation matrix tag->camera
                    R_ct = np.matrix(cv2.Rodrigues(rvec)[0])  # cv2.Rodrigues(rvec)
                    R_tc = R_ct.T

                    #-- Get the attitude in terms of euler 321 (Needs to be flipped first)
                    roll_marker, pitch_marker, yaw_marker = self._rotationMatrixToEulerAngles(R_tc) # self._R_flip*R_tc
                    # quaternion = self.euler_to_quaternion(yaw_marker, pitch_marker, roll_marker)
                    #print('quat: ', quaternion, ' euler: ', [roll_marker, pitch_marker, yaw_marker])

                    total_markers = range(0, marker_IDs.size)
                    for ids, corners, i in zip(marker_IDs, marker_corners, total_markers):

                        corners = corners.reshape(4, 2)
                        corners = corners.astype(int)

                        M = cv2.moments(corners)
                        cx = int(M['m10']/M['m00'])
                        cy = int(M['m01']/M['m00'])                
                        area = round(int(cv2.arcLength(corners,True))/marker_size, 1)
                        # print('cx, cy: ', cx, cy, area )
                                    
                        # depth of aligned depth frame
                        depth_in_meters = aligned_depth_frame.get_distance(cx, cy)
                        depth_point = rs.rs2_deproject_pixel_to_point(aligned_depth_intrinsic,[cx,cy], depth_in_meters)
                        v_cam = np.array([[depth_point[0]],[depth_point[1]],[depth_point[2]]]) 
                        # print("Camera Coordinate System: x=%.3lf, y=%.3lf, z=%.3lf\n" % (depth_point[2], depth_point[0], depth_point[1]))  
                           
                        global aruco_id, aruco_x, aruco_y, aruco_th 
                        aruco_id = float(ids[0])
                        aruco_x, aruco_y, aruco_th = self.estimate_pose(ids, yaw_marker, v_cam)
                                
                        if aruco_x != None and aruco_y != None and aruco_th != None:
                            # define region of interest
                            if (210 < cx < 550) and (3.7 < area < 75.8): 
                                print("id: "+str(ids[0])+": measured cx "+str(cx)+", cy "+str(cy)+" and area "+str(area)+", within good visible range. node will publish robot's estimated location.") 
                                self.pub_aruco_info()
                            else:
                                pass
                                
                        cv2.putText(color_image, f"Area:{area}", (cx-120,cy-80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,255,255), 1)
                        cv2.putText(color_image,f"{depth_in_meters}",(cx,cy),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,150,255),2)
                        cv2.putText(color_image, f"id: {ids[0]}", (30,30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 2)
                        # cv2.circle(color_image,(cx,cy),2,(0,0,255),-1) 
                        #       
                #########################################################
                #-------------------------------------------------------#
                #########################################################
     

                if show_video:
                    #--- Display the frame: Show images 
                    cv2.namedWindow('RealSense-RGB', cv2.WINDOW_AUTOSIZE)
                    cv2.imshow('RealSense-RGB', color_image)

                    #--- use 'q' to quit
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        cv2.destroyAllWindows()
                        break

        except RuntimeError:
            print("Azeez: Frame didn't arrive within 5000")
            ctx = rs.context()
            devices = ctx.query_devices()
            for dev in devices:
                dev.hardware_reset()
            self.rs_aruco_pose(pipeline, marker_size, show_video)
            time.sleep(1)
            return
 
        finally:
            cv2.destroyAllWindows()

            # Stop streaming
            pipeline.stop()


# -----------------------------------------
# -----------------------------------------
# -----------------------------------------

    def estimate_pose(self, ids, yaw_marker, v_cam):
        try: 
            if (str(ids[0]) in self.ids_to_find): # and (str(ids[0]) != self.current_detected_id):      
                coordinates = self.ids_to_find[str(ids[0])]
                x_bw = coordinates[0]
                y_bw = coordinates[1]
                t_bw = math.radians(coordinates[2])
                if (str(ids[0]) != self.current_detected_id): 
                    start_time = datetime.now().second
                    self.current_detected_id = str(ids[0])
                    list__xk = []
                    list__yk = []

                # rotation matrix 
                t_map_qr = [[cos(t_bw), -sin(t_bw), x_bw],
                            [sin(t_bw),  cos(t_bw), y_bw],
                            [0, 0, 1]] 
            
                rot_qr_cam = [[cos(yaw_marker), -sin(yaw_marker), 0],
                                [sin(yaw_marker), cos(yaw_marker), 0],
                                [0, 0, 1]] 
            
                trans_qr_cam = [[1, 0, v_cam[2]],
                                [0, 1, v_cam[0]], 
                                [0, 0, 1]] 
            
                t_map_cam = np.dot(np.dot(t_map_qr, rot_qr_cam), trans_qr_cam) 
                x_map_cam, y_map_cam, yaw_map_cam = t_map_cam[0][2], t_map_cam[1][2], math.atan2(t_map_cam[1][0], t_map_cam[0][0]); 
                #print('x_', x_map_cam, 'y_', y_map_cam, 't_', yaw_map_cam)
                
                list__xk.append(x_map_cam)
                list__yk.append(y_map_cam)

                if (0.9 < abs(datetime.now().second - start_time)):
                    
                    self.current_detected_id = 'none'

                    # Finding the IQR
                    percentile25x = np.quantile(list__xk, 0.25) 
                    percentile75x = np.quantile(list__xk, 0.75) 

                    percentile25y = np.quantile(list__yk, 0.25) 
                    percentile75y = np.quantile(list__yk, 0.75) 

                    # Finding upper and lower limit
                    iqr_x = percentile75x - percentile25x
                    upper_limitx = percentile75x + 1.5 * iqr_x
                    lower_limitx = percentile25x - 1.5 * iqr_x

                    iqr_y = percentile75y - percentile25y
                    upper_limity = percentile75y + 1.5 * iqr_y
                    lower_limity = percentile25y - 1.5 * iqr_y

                    # Trimming
                    inlier_list_x, inlier_list_y = [], []
                    for i in range(len(list__xk)):
                        if (list__xk[i] > lower_limitx) and (list__xk[i] < upper_limitx): 
                            inlier_list_x.append(list__xk[i])
                    mean_x = sum(inlier_list_x) / len(inlier_list_x)

                    for i in range(len(list__yk)):
                        if (list__yk[i] > lower_limity) and (list__yk[i] < upper_limity):
                            inlier_list_y.append(list__yk[i])
                    mean_y = sum(inlier_list_y) / len(inlier_list_y)

                    # Reset the lists to use them again
                    list__xk = []
                    list__yk = []


                    return float(mean_x), float(mean_y), float(yaw_map_cam) 
            else:
                return None, None, None # print("Nothing detected.")
            
        except ZeroDivisionError:
            return None, None, None # continue
        
        except ValueError:
            return None, None, None # continue         
        
# -----------------------------------------
# -----------------------------------------
# -----------------------------------------

def main(args=None):

  # Initialize the rclpy library
    rclpy.init(args=args)   
    try: 
        client_vel = Aruco_pub()
        rclpy.spin(client_vel)
    except:                        # except SystemExit: # <- process the exception 
        client_vel.destroy_node()  #    rclpy.logging.get_logger("Route_pub").info('Exited')
        rclpy.shutdown()
    
# -----------------------------------------
# -----------------------------------------
# -----------------------------------------

if __name__ == '__main__':
    main()












































        # # Load images
        # img1 = cv2.imread('image1.jpg',0)  # queryImage
        # img2 = cv2.imread('image2.jpg',0)  # trainImage

        # # Initiate ORB detector
        # orb = cv2.ORB_create()

        # # Find the keypoints and descriptors with ORB
        # kp1, des1 = orb.detectAndCompute(img1,None)
        # kp2, des2 = orb.detectAndCompute(img2,None)

        # # Create BFMatcher object
        # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # # Match descriptors
        # matches = bf.match(des1,des2)

        # # Sort them in the order of their distance
        # matches = sorted(matches, key = lambda x:x.distance)
        # # Draw first 10 matches
        # img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # cv2.imshow('Matches',img3)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # # Extract location of good matches
        # points1 = np.zeros((len(matches), 2), dtype=np.float32)
        # points2 = np.zeros((len(matches), 2), dtype=np.float32)
        # for i, match in enumerate(matches):
        #     points1[i, :] = kp1[match.queryIdx].pt
        #     points2[i, :] = kp2[match.trainIdx].pt

        # # Find homography
        # h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

        # # Define the distortion coefficients d
        # distCoef = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        # # Use homography to get intrinsic camera parameters
        # solutions = cv2.decomposeHomographyMat(h)
        # print('Camera matrix: ', solutions)
        # # Initialize lists to hold the parameters for each solution
        # fx_values = []
        # fy_values = []
        # ppx_values = []
        # ppy_values = []

        # # Loop over the solutions
        # for solution in solutions:
        #     R, t, n = solution

        #     # The camera matrix is not directly available in the solutions.
        #     # However, you can estimate fx, fy, ppx, ppy based on your specific camera model and setup.
        #     # Here, we're just printing the rotation and translation for each solution.
        #     # print('Rotation matrix: ', R)
        #     # print('Translation vector: ', t)

        #     # You can add your own code here to estimate fx, fy, ppx, ppy based on R and t.
        #     # For example, if you have a pinhole camera model, fx and fy might be related to the diagonal of R,
        #     # and ppx and ppy might be related to the last column of R.
        #     # These are just placeholders and might not be correct for your specific camera model and setup.
        #     fx = np.linalg.norm(R, axis=0).mean()
        #     fy = fx  # Assume square pixels
        #     ppx = R[0, 2]
        #     ppy = R[1, 2]

        #     # Append the estimated parameters to the lists
        #     fx_values.append(fx)
        #     fy_values.append(fy)
        #     ppx_values.append(ppx)
        #     ppy_values.append(ppy)