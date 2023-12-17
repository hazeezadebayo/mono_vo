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
import scipy.linalg
from copy import deepcopy
from threading import Lock



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






import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Quaternion, Vector3, TwistStamped, TransformStamped
from sensor_msgs.msg import Imu
from tf2_ros import TransformBroadcaster
import numpy as np
from scipy.linalg import sqrtm
from copy import deepcopy
import math

class UKFException(Exception):
    """Raise for errors in the UKF, usually due to bad inputs"""

class SensorFusionUKF(Node):
    def __init__(self):
        super().__init__('sensor_fusion_ukf')

        # Initialize UKF for sensor fusion
        self.state_estimator = self.init_ukf()

        # ROS2 Publishers
        self.fused_pose_pub = self.create_publisher(PoseStamped, 'fused_pose', 10)

        # ROS2 Subscribers
        self.camera_pose_sub = self.create_subscription(
            PoseStamped, 'camera_pose', self.camera_pose_callback, 10)
        
        self.imu_sub = self.create_subscription(
            Imu, 'imu_data', self.imu_callback, 10)

        # TF Broadcaster for fused transformation
        self.tf_broadcaster = TransformBroadcaster(self)

        image_raw_topic = '/kinect_camera/image_raw'
        depth_image_topic = '/kinect_camera/depth/image_raw'
        camera_info = '/kinect_camera/camera_info'

        self.bridge = CvBridge()
        self.trajectory = [] 
        self.traj = np.zeros((600,600,3), dtype=np.uint8)
        plt.figure()

        self.last_time = 0

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
    # -----------------------------------------

    def depth_image_callback(self, msg):
        cv_depthimage = self.bridge.imgmsg_to_cv2(msg, "32FC1")

    # -----------------------------------------
    # -----------------------------------------  

    def image_callback(self, msg):
        # cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8") # coloured image
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, "mono8") # grayscale image

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

    def init_ukf(self):
        num_states = 6  # Number of states (x, y, z, roll, pitch, yaw)
        process_noise = 0.001 * np.eye(num_states)  # Process noise covariance per unit time
        # process_noise = np.eye(6)
        # process_noise[0][0] = 0.0001    # x
        # process_noise[1][1] = 0.0001    # y
        # process_noise[2][2] = 0.0004    # theta or compass heading or yaw
        # process_noise[3][3] = 0.0025    # v or long. velocity
        # process_noise[4][4] = 0.0025    # yaw rate ____ v - v  angle about z from IMU
        # process_noise[5][5] = 0.0025    # long. acc from IMU
        initial_state = np.zeros(num_states)  # Initial values for the states
        initial_covar = 0.0001 * np.eye(num_states)  # Initial covariance matrix
        alpha = 0.04  # UKF tuning parameter
        k = 0.0  # UKF tuning parameter
        beta = 2.0  # UKF tuning parameter

        # Define the iterate function for prediction
        def iterate_function(state, timestep, inputs):
            ret = np.zeros(len(state))
            ret[0] = state[0] + timestep * state[3] * math.cos(state[2])  # x
            ret[1] = state[1] + timestep * state[3] * math.sin(state[2])  # y
            ret[2] = state[2] + timestep * state[4]  # yaw
            ret[3] = state[3] + timestep * state[5]  # long. velocity
            ret[4] = state[4]  # yaw rate i.e theta dot
            ret[5] = state[5]  # long. acceleration
            return ret

        # pass all the parameters into the UKF!
        # number of state variables, process noise, initial state, initial coariance, three tuning paramters, and the iterate function
        return UKF(num_states, process_noise, initial_state, initial_covar,
                   alpha, k, beta, iterate_function)

    # -----------------------------------------
    # -----------------------------------------

    # def camera_pose_callback(self, msg):
    #     # Update the state estimator with camera pose data
    #     camera_data = np.array([msg.pose.position.x, msg.pose.position.y,
    #                             msg.pose.position.z])
    #     self.state_estimator.update(list(range(3)), camera_data, np.eye(3) * 0.01)
        
    # def imu_callback(self, msg):
    #     # Update the state estimator with IMU data
    #     imu_data = np.array([msg.angular_velocity.x, msg.angular_velocity.y,
    #                          msg.angular_velocity.z])
    #     self.state_estimator.update(list(range(3, 6)), imu_data, np.eye(3) * 0.01)

    def camera_pose_callback(self, msg):
        # TODO: Process camera pose data
        # Update the state estimator with camera pose data
        camera_data = np.array([msg.pose.position.x, msg.pose.position.y,
                                msg.pose.position.z,
                                msg.pose.orientation.roll, msg.pose.orientation.pitch,
                                msg.pose.orientation.yaw])
        self.state_estimator.update(list(range(6)), camera_data, np.eye(6) * 0.01)
        print ("--------------------------------------------------------")
        print ("Real state: ", camera_data)
        # updating isn't bad either
        # create measurement noise covariance matrices
        # r_imu = np.zeros([2, 2]) #supplies acc. and yaw_rate 
        # r_imu[0][0] = 0.01
        # r_imu[1][1] = 0.03
        # r_compass = np.zeros([1, 1]) # for 2
        # r_compass[0][0] = 0.02
        # r_encoder = np.zeros([2, 2]) #no longer 3 gives 0,1
        # r_encoder[0][0] = 0.001
        # r_encoder[1][1] = 0.001
        # remember that the updated states should be zero-indexed
        # the states should also be in the order of the noise and data matrices
        # imu_data = np.array([imu_yaw_rate, imu_accel])
        # compass_data = np.array([compass_hdg])
        # encoder_data = np.array([encoder_x, encoder_y])
        # state_estimator.update([4, 5], imu_data, r_imu) #  gyro ad accelero
        # state_estimator.update([2], compass_data, r_compass)  #theta
        # state_estimator.update([3], encoder_vel, r_encoder)
        # state_estimator.update([0, 1], encoder_data, r_encoder) # x, y

    # -----------------------------------------
    # -----------------------------------------

    def imu_callback(self, msg):
        # TODO: Process IMU data
        # Update the state estimator with IMU data
        imu_data = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y,
                             msg.linear_acceleration.z,
                             msg.angular_velocity.roll, msg.angular_velocity.pitch,
                             msg.angular_velocity.yaw])
        self.state_estimator.update(list(range(6)), imu_data, np.eye(6) * 0.01)

    # -----------------------------------------
    # -----------------------------------------

    def filter_process(self):
        # TODO: Implement the fusion process
        # Get the fused state from the UKF and publish the result
        # Also, broadcast the fused transformation as a TF frame

        # Perform the prediction step
        timestep = 0.01  # Placeholder value, replace with actual timestep
        self.state_estimator.predict(timestep)

        # Get the fused state
        fused_state = self.state_estimator.get_state()
        fused_covariance = self.state_estimator.get_covariance()

        # Publish the fused pose
        fused_pose_msg = PoseStamped()
        fused_pose_msg.pose.position.x = fused_state[0]
        fused_pose_msg.pose.position.y = fused_state[1]
        fused_pose_msg.pose.position.z = fused_state[2]
        # Assuming the state contains roll, pitch, and yaw in the last three elements
        fused_pose_msg.pose.orientation = Quaternion(
            roll=fused_state[3], pitch=fused_state[4], yaw=fused_state[5])
        self.fused_pose_pub.publish(fused_pose_msg)

        # Broadcast the fused transformation as a TF frame
        tf_msg = TransformStamped()
        tf_msg.header.stamp = self.get_clock().now().to_msg()
        tf_msg.header.frame_id = 'world'
        tf_msg.child_frame_id = 'robot_frame'  # Replace with your robot frame ID
        tf_msg.transform.translation = Vector3(x=fused_state[0], y=fused_state[1], z=fused_state[2])
        tf_msg.transform.rotation = Quaternion(
            roll=fused_state[3], pitch=fused_state[4], yaw=fused_state[5])
        self.tf_broadcaster.sendTransform(tf_msg)

        # Print the fused state for debugging
        self.get_logger().info(f"Fused State: {fused_state}")
        print ("--------------------------------------------------------")
        print ("Estimated state: ", fused_state )

# -----------------------------------------
# -----------------------------------------

class UKF:
    def __init__(self, num_states, process_noise, initial_state, initial_covar, alpha, k, beta, iterate_function):
        self.n_dim = num_states
        self.n_sig = self.n_dim*2+1
        self.q = process_noise
        self.x = initial_state
        self.p = initial_covar
        self.beta = beta
        self.alpha = alpha
        self.k = k
        self.iterate = iterate_function

        self.lambd = pow(self.alpha, 2) * (self.n_dim + self.k) - self.n_dim
        self.covar_weights = np.zeros(self.n_sig)
        self.mean_weights = np.zeros(self.n_sig)

        self.covar_weights[0] = (self.lambd / (self.n_dim + self.lambd)) + (1 - pow(self.alpha, 2) + self.beta)
        self.mean_weights[0] = (self.lambd / (self.n_dim + self.lambd))

        for i in range(1, self.n_sig):
            self.covar_weights[i] = 1 / (2*(self.n_dim + self.lambd))
            self.mean_weights[i] = 1 / (2*(self.n_dim + self.lambd))

        self.sigmas = self.__get_sigmas()
        self.lock = Lock()

    # -----------------------------------------
    # -----------------------------------------

    def get_state(self):
        return self.x
    
    # -----------------------------------------
    # -----------------------------------------

    def get_covariance(self):
        return self.p

    # -----------------------------------------
    # -----------------------------------------

    def __get_sigmas(self):
        """generates sigma points"""
        ret = np.zeros((self.n_dim, self.n_sig))
        spr_mat = sqrtm((self.n_dim + self.lambd)*self.p)
        ret[:, 1:self.n_dim+1] = self.x + spr_mat
        ret[:, self.n_dim+1:] = self.x - spr_mat
        return ret

    # -----------------------------------------
    # -----------------------------------------

    def set_state(self, state):
        self.x = state
        self.sigmas = self.__get_sigmas()

    # -----------------------------------------
    # -----------------------------------------

    def predict(self, dt):
        self.lock.acquire()
        self.sigmas = self.iterate(self.sigmas, dt)
        self.x = np.mean(self.sigmas, axis=1)
        self.p = np.cov(self.sigmas)
        self.lock.release()

    # -----------------------------------------
    # -----------------------------------------

    def update(self, states, data, r_matrix):
        self.lock.acquire()
        num_states = len(states)
        # create y, sigmas of just the states that are being updated
        sigmas_split = np.split(self.sigmas, self.n_dim)
        y = np.concatenate([sigmas_split[i] for i in states])
        # create y_mean, the mean of just the states that are being updated
        x_split = np.split(self.x, self.n_dim)
        y_mean = np.concatenate([x_split[i] for i in states])
        # differences in y from y mean
        y_diff = deepcopy(y)
        x_diff = deepcopy(self.sigmas)
        for i in range(self.n_sig):
            for j in range(num_states):
                y_diff[j][i] -= y_mean[j]
            for j in range(self.n_dim):
                x_diff[j][i] -= self.x[j]
        # covariance of measurement
        p_yy = np.zeros((num_states, num_states))
        for i, val in enumerate(np.array_split(y_diff, self.n_sig, 1)):
            p_yy += self.covar_weights[i] * val.dot(val.T)
        # add measurement noise
        p_yy += r_matrix
        # covariance of measurement with states
        p_xy = np.zeros((self.n_dim, num_states))
        for i, val in enumerate(zip(np.array_split(y_diff, self.n_sig, 1), np.array_split(x_diff, self.n_sig, 1))):
            p_xy += self.covar_weights[i] * val[1].dot(val[0].T)

        k = np.dot(p_xy, np.linalg.inv(p_yy))

        y_actual = data

        self.x += np.dot(k, (y_actual - y_mean))
        self.p -= np.dot(k, np.dot(p_yy, k.T))
        self.sigmas = self.__get_sigmas()

        self.lock.release()

    # -----------------------------------------
    # -----------------------------------------

def main(args=None):
    rclpy.init(args=args)
    try: 
        sensor_fusion_node = SensorFusionUKF()
        rclpy.spin(sensor_fusion_node)
    except:   # except SystemExit: # <- process the exception 
        sensor_fusion_node.destroy_node()  #    rclpy.logging.get_logger("Route_pub").info('Exited')
        rclpy.shutdown()

# -----------------------------------------
# -----------------------------------------

if __name__ == '__main__':
    main()














    # def predict(self, timestep, inputs=[]):
    #     """
    #     performs a prediction step
    #     :param timestep: float, amount of time since last prediction
    #     """

    #     self.lock.acquire()

    #     sigmas_out = np.array([self.iterate(x, timestep, inputs) for x in self.sigmas.T]).T

    #     x_out = np.zeros(self.n_dim)

    #     # for each variable in X
    #     for i in range(self.n_dim):
    #         # the mean of that variable is the sum of
    #         # the weighted values of that variable for each iterated sigma point
    #         x_out[i] = sum((self.mean_weights[j] * sigmas_out[i][j] for j in range(self.n_sig)))

    #     p_out = np.zeros((self.n_dim, self.n_dim))
    #     # for each sigma point
    #     for i in range(self.n_sig):
    #         # take the distance from the mean
    #         # make it a covariance by multiplying by the transpose
    #         # weight it using the calculated weighting factor
    #         # and sum
    #         diff = sigmas_out.T[i] - x_out
    #         diff = np.atleast_2d(diff)
    #         p_out += self.covar_weights[i] * np.dot(diff.T, diff)

    #     # add process noise
    #     p_out += timestep * self.q

    #     self.sigmas = sigmas_out
    #     self.x = x_out
    #     self.p = p_out

    #     self.lock.release()
