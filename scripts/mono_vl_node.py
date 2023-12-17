#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2, math, sys, warnings
import numpy as np
import matplotlib.pyplot as plt


from datetime import datetime
import time
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

# [image_subscriber]: Camera Info: sensor_msgs.msg.CameraInfo(header=std_msgs.msg.Header(stamp=builtin_interfaces.msg.Time(sec=38, nanosec=736000000), 
# frame_id='kinect_camera_optical'), 
# height=480, width=640, 
# distortion_model='plumb_bob', 
# d=[0.0, 0.0, 0.0, 0.0, 0.0], 
# k=array([528.43375656,   0.        , 320.5       ,   0.        ,
#        528.43375656, 240.5       ,   0.        ,   0.        ,
#          1.        ]), 
# r=array([1., 0., 0., 0., 1., 0., 0., 0., 1.]), 
# p=array([528.43375656,   0.        , 320.5       ,  -0.        ,
#          0.        , 528.43375656, 240.5       ,   0.        ,
#          0.        ,   0.        ,   1.        ,   0.        ]), 
# binning_x=0, binning_y=0, 
# roi=sensor_msgs.msg.RegionOfInterest(x_offset=0, y_offset=0, height=0, width=0, do_rectify=False))

    # could be replaced with alpha = old depth / new depth of any pixel u = (u,v)', where alpha is the scale
    # or
    # sample coordinates x, y of (t) and x, y of (t-1) from the odom when picture is snapped and use them directly below
    # or
    # that xyz - xyz
    # or
    # that T (x y z) from rt left camera relative to right --- although i highly doubt this. it might work sha

    #def getAbsoluteScale(self, frame_id):
    #    """ Obtains the absolute scale utilizing
    #    the ground truth poses. (KITTI dataset)"""
    #    z_prev / z
    #    return np.sqrt((x - x_prev) * (x - x_prev) + (y - y_prev) * (y - y_prev) + (z - z_prev) * (z - z_prev))




# pip3 uninstall opencv-contrib-python
# pip3 install opencv-contrib-python

# source install/setup.sh
# source /opt/ros/humble/setup.bash
# ros2 run mono_vo mono_vo_node.py











class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')

        image_raw_topic = '/kinect_camera/image_raw'
        depth_image_topic = '/kinect_camera/depth/image_raw'
        camera_info = '/kinect_camera/camera_info'

        self.bridge = CvBridge()

        self.cv_image = None
        self.prev_keypoint = None
        self.prev_image = None
        self.pp = None # principal point of the camera #  (image_width / 2, image_height / 2).
        self.focal_length = None
        self.trajectory = [] 
        self.traj = np.zeros((600,600,3), dtype=np.uint8)
        plt.figure()

        self.marker_size = 13.5 # [-cm]  # should be metre so that the v_cam estimate can be in real world metres.
        self.current_detected_id = 'none'
        self.aruco_stat_time = time.time_ns()
        self.show_video =  True
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

        self.image_subscription = self.create_subscription(
            Image,
            image_raw_topic,
            self.image_callback,
            10)

        self.depth_image_subscription = self.create_subscription(
            Image,
            depth_image_topic,
            self.depth_image_callback,
            10)

        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            camera_info,
            self.camera_info_callback,
            10)
    
    # -----------------------------------------
    # -----------------------------------------  

    def image_callback(self, msg):
        # cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8") # coloured image
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, "mono8") # grayscale image

    # -----------------------------------------
    # -----------------------------------------  

    def plot_trajectory(self):
        # Unpack the trajectory points
        x, y = zip(*self.trajectory)
        # Clear the current figure's content
        plt.clf()
        # Create the plot
        plt.plot(x, y, 'ro-')
        plt.title('Robot Trajectory')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True)
        # Draw the plot and pause for a short period
        plt.draw()
        plt.pause(0.001)


    # -----------------------------------------
    # -----------------------------------------

    def euler_to_quaternion(self, yaw, pitch, roll):
        qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
        qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
        qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        return [qx, qy, qz, qw]

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

    def estimate_camera_prop(self, img1):
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
        return cam_mat, distCoef

    # -----------------------------------------
    # -----------------------------------------

    def depth_image_callback(self, msg):
        cv_depthimage = self.bridge.imgmsg_to_cv2(msg, "32FC1")

        marker_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_250)
        param_markers = cv2.aruco.DetectorParameters_create()
                
        try:
            # gray_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY) # if self.cv_image is coloured
            gray_image = self.cv_image

            cv2.line(gray_image, (280,0),(280,480),(255,255,0),1)
            cv2.line(gray_image, (400,0),(400,480),(255,255,0),1)
                  
            #########################################################
            #-------------------- ARUCO LOCALIZER ------------------#
            #########################################################
            # marker_corners, marker_IDs, reject = cv2.aruco.detectMarkers(gray_image, marker_dict, parameters=param_markers )
            marker_corners, marker_IDs, reject = cv2.aruco.detectMarkers(
                            image= gray_image, 
                            dictionary= marker_dict, 
                            parameters= param_markers,
                            cameraMatrix= self.camera_matrix, 
                            distCoeff= self.dist_coeff)

            if marker_corners:
                #camera to marker rotation and translation.
                ret = cv2.aruco.estimatePoseSingleMarkers(marker_corners, self.marker_size, self.camera_matrix, self.dist_coeff)
                rvec, tvec = ret[0][0,0,:], ret[1][0,0,:]

                #-- Draw the detected marker and put a reference frame over it
                cv2.aruco.drawDetectedMarkers(gray_image, marker_corners)
                cv2.aruco.drawAxis(gray_image, self.camera_matrix, self.dist_coeff, rvec, tvec, 10)

                #-- Obtain the rotation matrix tag->camera
                R_ct = np.matrix(cv2.Rodrigues(rvec)[0])  # cv2.Rodrigues(rvec)
                R_tc = R_ct.T

                #-- Get the attitude in terms of euler 321 (Needs to be flipped first)
                roll_marker, pitch_marker, yaw_marker = self._rotationMatrixToEulerAngles(R_tc) # self._R_flip*R_tc
                # quaternion = self.euler_to_quaternion(yaw_marker, pitch_marker, roll_marker)
                # print('quat: ', quaternion, ' euler: ', [roll_marker, pitch_marker, yaw_marker])

                total_markers = range(0, marker_IDs.size)
                for ids, corners, i in zip(marker_IDs, marker_corners, total_markers):

                    corners = corners.reshape(4, 2)
                    corners = corners.astype(int)

                    M = cv2.moments(corners)
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])                
                    area = round(int(cv2.arcLength(corners,True))/self.marker_size, 1)
                    # print('cx, cy: ', cx, cy, area )

                    # Get the depth value at (cx, cy)
                    depth_in_meters = cv_depthimage[cy, cx]
                    # Calculate in real-world camera coordinates | self.camera_matrix = np.array([[fx, 0, ppx], [0, fy, ppy], [0, 0, 1]]) | intrinsic matrix
                    x = (cx - self.camera_matrix[0, 2]) * depth_in_meters / self.camera_matrix[0, 0] 
                    y = (cy - self.camera_matrix[1, 2]) * depth_in_meters / self.camera_matrix[1, 1]
                    z = depth_in_meters
                    v_cam = np.array([x, y, z])  # z is forward though, in camera frame
                    # print("Camera Coordinate System: x=%.3lf, y=%.3lf, z=%.3lf\n" % (z, x, y))  

                    # ideally i do not have to use a depth camera for this and i could use the below code to estimate 
                    # the relative position of the camera with the qr marker. this only works if marker size is specified in metres/real world units.
                    # say we dont have a depth camera and we just want to estimate x, y from the image itself without real world x y
                    # -- Now get Position and attitude of the camera respect to the marker
                    # v_cam = -R_tc*np.matrix(tvec).T # rotate and translate the position of the marker (given by tvec) from the camera frame to the marker frame
                    # print("Camera Coordinate System: x=%.3lf, y=%.3lf, z=%.3lf\n" % (v_cam[2], v_cam[0], v_cam[1]))  
                    
                    global aruco_id, aruco_x, aruco_y, aruco_th 
                    aruco_id = float(ids[0])
                    aruco_x, aruco_y, aruco_th = self.estimate_pose(ids, yaw_marker, v_cam)
                            
                    if aruco_x != None and aruco_y != None and aruco_th != None:
                        # define region of interest
                        if (210 < cx < 550) and (3.7 < area < 75.8): 
                            print("id: "+str(ids[0])+": measured cx "+str(cx)+", cy "+str(cy)+" and area "+str(area)+", within good visible range. node will publish robot's estimated location.") 
                        else:
                            pass
                            
                    cv2.putText(gray_image, f"Area:{area}", (cx-120,cy-80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,255,255), 1)
                    cv2.putText(gray_image,f"{depth_in_meters}",(cx,cy),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,150,255),2)
                    cv2.putText(gray_image, f"id: {ids[0]}", (30,30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 2)
                    # cv2.circle(color_image,(cx,cy),2,(0,0,255),-1) 
     
            #########################################################
            #-------------------------------------------------------#
            #########################################################
    
            if self.show_video: # Display the frame: Show images 
                # cv2.namedWindow('RealSense-RGB', cv2.WINDOW_AUTOSIZE)
                cv2.imshow("Depth Image Window", cv_depthimage)
                #--- use 'q' to quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cv2.destroyAllWindows()

        except RuntimeError:
            return
 
        finally:
            cv2.destroyAllWindows()

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
                t_map_qr = [[math.cos(t_bw), -math.sin(t_bw), x_bw],
                            [math.sin(t_bw),  math.cos(t_bw), y_bw],
                            [0, 0, 1]] 
            
                rot_qr_cam = [[math.cos(yaw_marker), -math.sin(yaw_marker), 0],
                                [math.sin(yaw_marker), math.cos(yaw_marker), 0],
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

    def camera_info_callback(self, msg):
        # self.get_logger().info('Camera Info: %s' % msg)
        # Focal lengths in pixel coordinates
        fx = msg.k[0]  # Focal length in x direction
        fy = msg.k[4]  # Focal length in y direction
        # Principal point
        ppx = msg.k[2]  # x coordinate of principal point
        ppy = msg.k[5]  # y coordinate of principal point
        # Distortion coefficients
        self.dist_coeff = np.array(msg.d)  # Distortion coefficients
        # Image dimensions
        image_height = msg.height  # Image height
        image_width = msg.width  # Image width
        # self.get_logger().info('Camera Info: fx=%f, fy=%f, \
        #                        ppx=%f, ppy=%f, distortion_coeff=%s, \
        #                        image_height=%d, image_width=%d' \
        #                        % (fx, fy, ppx, ppy, distortion_coeff, image_height, image_width))
        # if self.focal_length is None and self.pp is None:
        self.camera_matrix = np.array([[fx, 0, ppx], [0, fy, ppy], [0, 0, 1]])
        # self.pp = (ppx, ppy) # (ppx + ppy) / 2 # principal point of the camera #  (image_width / 2, image_height / 2).
        # self.focal_length = (fx + fy) / 2 

    # -----------------------------------------
    # -----------------------------------------  

def main(args=None):
    rclpy.init(args=args)
    try: 
        image_subscriber = ImageSubscriber()
        rclpy.spin(image_subscriber)
    except:   # except SystemExit: # <- process the exception 
        image_subscriber.destroy_node()  #    rclpy.logging.get_logger("Route_pub").info('Exited')
        rclpy.shutdown()


if __name__ == '__main__':
    main()


























# device_serial_no = '843112072968'
        # if device_serial_no != 'none':
        #     ctx = rs.context()
        #     if len(ctx.devices) > 0:
        #         for d in ctx.devices:
        #             print ('Found device: ', \
        #                     d.get_info(rs.camera_info.name), ' ', \
        #                     d.get_info(rs.camera_info.serial_number))

        #             if device_serial_no != d.get_info(rs.camera_info.serial_number):

        #                 show_video = True

        #                 # define pipeline
        #                 pipeline = rs.pipeline()
        #                 config = rs.config()

        #                 config.enable_device(d.get_info(rs.camera_info.serial_number)) 

        #                 config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
        #                 config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
                        
        #                 # Start streaming
        #                 pipeline.start(config)

        #                 self.rs_aruco_pose(pipeline, marker_size, show_video)

        #     else:
        #         print("No Intel Device connected")
        # else:
        #     self.generic_aruco_pose(pipeline, marker_size, show_video)


        # except RuntimeError:
        #     print("Azeez: Frame didn't arrive within 5000")
        #     ctx = rs.context()
        #     devices = ctx.query_devices()
        #     for dev in devices:
        #         dev.hardware_reset()
        #     self.rs_aruco_pose(pipeline, marker_size, show_video)
        #     time.sleep(1)
        #     return

# import spatial
# def removeDuplicates(points, threshold=30):
#     """Remove duplicate points that are within a certain Euclidean distance from each other."""
#     tree = spatial.KDTree(points)
#     groups = list(tree.query_ball_tree(tree, threshold))
#     new_points = []
#     for group in groups:
#         new_points.append(np.mean(points[group], axis=0))
#     return np.array(new_points)


# def deRotatePatch(patch):
#     # Compute the Harris matrix
#     harris_matrix = cv2.cornerHarris(patch, 2, 3, 0.04)
#     # Compute the eigenvalues and eigenvectors of the Harris matrix
#     _, _, eigenvectors = np.linalg.svd(harris_matrix)
#     # Find the direction of the most dominant gradient
#     dominant_direction = eigenvectors[0]
#     # Compute the angle to rotate the patch
#     angle = np.arctan2(dominant_direction[1], dominant_direction[0])
#     # Create a rotation matrix
#     rotation_matrix = cv2.getRotationMatrix2D((patch.shape[1]/2, patch.shape[0]/2), angle, 1)
#     # De-rotate the patch
#     de_rotated_patch = cv2.warpAffine(patch, rotation_matrix, (patch.shape[1], patch.shape[0]), flags=cv2.INTER_CUBIC)
#     return de_rotated_patch
    
        # self.T_vectors.append(tuple([[0], [0], [0]]))
        # self.R_matrices.append(tuple(np.zeros((3, 3))))
    
# self.tracker_type = 'Farneback' # 'Farneback' | 'KLT'
# def Farneback_featureTracking(prev_img, cur_img, prev_pts):
#     farneback_params = dict(pyr_scale=0.5, levels=5, winsize=13, iterations=10, poly_n=5, poly_sigma=1.1, flags=0)
#     flow = cv2.calcOpticalFlowFarneback(prev_img, cur_img, None, **farneback_params)
#     indices_yx = prev_pts.astype(int)
#     valid_indices = (
#         (indices_yx[:, 0] >= 0) & (indices_yx[:, 0] < flow.shape[0]) & (indices_yx[:, 1] >= 0) & (indices_yx[:, 1] < flow.shape[1])
#     )
#     flow_pts = flow[indices_yx[valid_indices, 0], indices_yx[valid_indices, 1]]
#     cur_pts = prev_pts[valid_indices] + flow_pts
#     px_diff = np.linalg.norm(prev_pts[valid_indices] - cur_pts, axis=1)
#     diff_mean = np.mean(px_diff)
#     return prev_pts[valid_indices], cur_pts, diff_mean

        # if self.tracker_type == 'KLT':
        #     self.px_ref, self.px_cur, px_diff = KLT_featureTracking(prev_img, cur_img, self.px_ref)
        # elif self.tracker_type == 'Farneback':
        #      self.px_ref, self.px_cur, px_diff = Farneback_featureTracking(prev_img, cur_img, self.px_ref)
        # else:
        #     raise ValueError(f"Unknown tracker type: {self.tracker_type}")
    
# def matching(matcher_type, description1, description2):
#     if matcher_type == 'BFMATCHER':
#         bf = cv2.BFMatcher()
#         matches = bf.knnMatch(description1, description2, k=2)
#     elif matcher_type == 'FLANN':
#         FLANN_INDEX_KDTREE = 1
#         index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
#         search_params = dict(checks=50)
#         flann = cv2.FlannBasedMatcher(index_params, search_params)
#         matches = flann.knnMatch(description1, description2, k=2)
#     else:
#         raise ValueError(f"Unknown matcher type: {matcher_type}")
#     return matches
