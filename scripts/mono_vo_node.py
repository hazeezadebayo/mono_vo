#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2, math
import numpy as np
import matplotlib.pyplot as plt






# [image_subscriber]: Camera Info: sensor_msgs.msg.CameraInfo(header=std_msgs.msg.Header(stamp=builtin_interfaces.msg.Time(sec=38, nanosec=736000000), 
# frame_id='kinect_camera_optical'), 
# height=480, width=640, 
# distortion_model='plumb_bob', d=[0.0, 0.0, 0.0, 0.0, 0.0], 
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






# CONSTANT VARIABLES
STAGE_FIRST_FRAME = 0  # The three STAGE variables
STAGE_SECOND_FRAME = 1  # define which function will be
STAGE_DEFAULT_FRAME = 2  # used in the update function.
# Parameters used for cv2.goodFeaturesToTrack (Shi-Tomasi Features)
feature_params = dict(maxCorners=500,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)
kMinNumFeature = 1000 # 1000  # Minimum amount of features needed, if less feature detection is used
fMATCHING_DIFF = 1.0 # 1.0  # Minimum difference in the KLT point correspondence
lk_params = dict(winSize=(25, 25), # Parameters used for cv2.calcOpticalFlowPyrLK (KLT tracker) | winSize=(21, 21) 
                 maxLevel=3, # termination cond. means the algorithm stops either after 30 iterations or when the error becomes less than 0.032.
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 0.01)) # 0.01





def drawMatchField(new_frame, px_ref, px_cur, F):
    # ----------------------------------------
    # Visualize epipolar lines and matches
    img_epilines = cv2.computeCorrespondEpilines(px_cur.reshape(-1, 1, 2), 2, F)
    img_epilines = img_epilines.reshape(-1, 3)
    img_lines = cv2.computeCorrespondEpilines(px_ref.reshape(-1, 1, 2), 1, F)
    img_lines = img_lines.reshape(-1, 3)
    for r, pt1, pt2 in zip(img_epilines, px_ref, px_cur):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [new_frame.shape[1], -(r[2] + r[0] * new_frame.shape[1]) / r[1]])
        # Ensure that pt1 and pt2 are tuples of integers
        pt1 = tuple(map(int, pt1))
        pt2 = tuple(map(int, pt2))
        cv2.line(new_frame, (x0, y0), (x1, y1), color, 1)
        cv2.circle(new_frame, pt1, 5, color, -1)
        cv2.circle(new_frame, pt2, 5, color, -1)
    # cv2.imshow('Epilines and Matches', new_frame)
    # cv2.waitKey(1)
    # cv2.destroyAllWindows()
    # ----------------------------------------


def removeDuplicates(points, threshold=30):
    """Remove duplicate points that are within a certain Euclidean distance from each other."""
    new_points = []
    for i in range(len(points)):
        # Compute distances to all other points
        distances = np.sqrt(np.sum((points - points[i])**2, axis=1))
        # Find points that are within the threshold distance
        close_points = points[distances < threshold]
        # Compute the mean of the close points
        new_point = np.mean(close_points, axis=0)
        new_points.append(new_point)
    return np.array(new_points)


def deRotateImage(image):
    # Resize the image to a smaller size
    # image = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2))
    # Compute the Harris matrix
    harris_matrix = cv2.cornerHarris(image, 2, 3, 0.04)
    # Compute the eigenvalues and eigenvectors of the Harris matrix
    _, _, eigenvectors = np.linalg.svd(harris_matrix)
    # Find the direction of the most dominant gradient
    dominant_direction = eigenvectors[0]
    # Compute the angle to rotate the image
    angle = np.arctan2(dominant_direction[1], dominant_direction[0])
    # Create a rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((image.shape[1]/2, image.shape[0]/2), angle, 1)
    # De-rotate the image
    de_rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)
    return de_rotated_image


def KLT_featureTracking(image_ref, image_cur, px_ref):
    kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, **lk_params)
    kp1, st, err = cv2.calcOpticalFlowPyrLK(image_cur, image_ref, kp2, None, **lk_params)

    d = np.abs(px_ref - kp1).reshape(-1, 2).max(-1)
    good = d < fMATCHING_DIFF
    print("num of good features found:", np.count_nonzero(good))

    if np.count_nonzero(good) == 0:
        print("Error: No matches were made.")
    elif np.count_nonzero(good) <= 5:
        print("Warning: No match was good. Returns the list without good point correspondence.")
        return kp1, kp2, 100000

    n_kp1, n_kp2 = kp1[good], kp2[good]

    diff_mean = np.mean(np.linalg.norm(n_kp1 - n_kp2, axis=1))
    return n_kp1, n_kp2, diff_mean


def betterMatches(F, points1, points2):
    points1 = points1.reshape(1, -1, 2)
    points2 = points2.reshape(1, -1, 2)
    newPoints1, newPoints2 = cv2.correctMatches(F, points1, points2)
    return newPoints1[0], newPoints2[0]


def filterMatches(matches, points1, points2):
    ratio = 0.7
    good_matches = [m for m, n in matches if m.distance < ratio * n.distance]
    good_points1 = np.float32([points1[m.queryIdx] for m in good_matches])
    good_points2 = np.float32([points2[m.trainIdx] for m in good_matches])
    return good_points1, good_points2


class VisualOdometry:
    def __init__(self, CameraIntrinMat):
        self.frame_stage = 0        # The current stage of the algorithm
        self.new_frame = None       # The current frame
        self.last_frame = None      # The previous frame
        self.skip_frame = False     # Determines if the next frame will be skipped
        self.cur_R = None           # The current concatenated Rotation Matrix
        self.cur_t = [0., 0., 0.]   # The current concatenated Translation Vector
        self.px_ref = None          # The previous corresponded feature points
        self.px_cur = None          # The current corresponded feature points
        self.K = CameraIntrinMat    # Camera Intrinsic Matrix
        self.Scale = 0              # Scale, used to scale the translation and rotation matrix
        self.new_cloud = None       # 3-D point cloud of current frame, i, and previous frame, i-1
        self.last_cloud = None      # 3-D point cloud of previous frame, i-1, and the one before that, i-2
        self.F_detectors = {'FAST': cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True),
                            'SIFT': cv2.SIFT_create(),     
                            'ORB': cv2.ORB_create(),   
                            # 'KAZE': cv2.KAZE_create(),
                            'SHI-TOMASI': 'SHI-TOMASI'}  # Dictionary of Feature Detectors available
        self.detector = self.F_detectors["FAST"] # The chosen Feature Detector
        self.matcher_type='BFMATCHER' # 'BFMATCHER' | 'FLANN'
        if self.matcher_type == 'BFMATCHER':
            self.matcher = cv2.BFMatcher()
            # self.matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        elif self.matcher_type == 'FLANN':
            FLANN_INDEX_KDTREE = 6
            search_params = dict(checks=100) # 50
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, table_number=6, key_size=12, multi_probe_level=2) # dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)


    def triangulatePoints(self, R, t):
        """Triangulates the feature correspondence points with
        the camera intrinsic matrix, rotation matrix, and translation vector.
        It creates projection matrices for the triangulation process."""
        # The canonical matrix (set as the origin)
        P0 = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0]])
        P0 = self.K.dot(P0)
        # Rotated and translated using P0 as the reference point
        P1 = np.hstack((R, t))
        P1 = self.K.dot(P1)
        # Reshaped the point correspondence arrays to cv2.triangulatePoints's format
        point1 = self.px_ref.reshape(2, -1)
        point2 = self.px_cur.reshape(2, -1)
        return cv2.triangulatePoints(P0, P1, point1, point2).reshape(-1, 4)[:, :3]


    def getRelativeScale(self):
        """ Returns the relative scale based on the 3-D point clouds
         produced by the triangulation_3D function. Using a pair of 3-D corresponding points
         the distance between them is calculated. This distance is then divided by the
         corresponding points' distance in another point cloud."""
        min_idx = min([self.new_cloud.shape[0], self.last_cloud.shape[0]])
        ratios = []  # List to obtain all the ratios of the distances
        for i in range(min_idx):
            if i > 0:
                Xk = self.new_cloud[i]
                p_Xk = self.new_cloud[i - 1]
                Xk_1 = self.last_cloud[i]
                p_Xk_1 = self.last_cloud[i - 1]

                if np.linalg.norm(p_Xk - Xk) != 0:
                    ratios.append(np.linalg.norm(p_Xk_1 - Xk_1) / np.linalg.norm(p_Xk - Xk))

        d_ratio = np.median(ratios) # Take the median of ratios list as the final ratio
        return d_ratio


    def frame_Skip(self, pixel_diff):
        """Determines if the current frame needs to be skipped.
         A frame is skipped on the basis that the current feature points
         are almost identical to the previous feature points, meaning the image
         was probably taken from the same place and the translation should be zero."""
        # We tried this parameter with 20, 15, 10, 5, 3, 2, 1 and 0
        # for one dataset and found that 3 produces the best results.
        return pixel_diff < 3


    def detectNewFeatures(self, cur_img):
        """Detects new features in the current frame.
        Uses the Feature Detector selected."""
        if self.detector == 'SHI-TOMASI':
            feature_pts = cv2.goodFeaturesToTrack(cur_img, **feature_params)
            feature_pts = np.array([x for x in feature_pts], dtype=np.float32).reshape((-1, 2))
        else:
            keypoints = self.detector.detect(cur_img, None)
            feature_pts = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        return feature_pts


    def processFirstFrame(self):
        """Process the first frame. Detects feature points on the first frame
        in order to provide them to the Kanade-Lucas-Tomasi Tracker"""
        self.px_ref = self.detectNewFeatures(self.new_frame)
        self.frame_stage = STAGE_SECOND_FRAME


    def processSecondFrame(self):
        """Process the second frame. Detects feature correspondence between the first frame
        and the second frame with the Kanade-Lucas-Tomasi Tracker. Initializes the
        rotation matrix and translation vector. The first point cloud is formulated."""
        # The images or roi used for the VO process (feature detection and tracking)
        prev_img, cur_img = self.last_frame, self.new_frame
        # Obtain feature correspondence points
        self.px_ref, self.px_cur, _diff = KLT_featureTracking(prev_img, cur_img, self.px_ref)
        # Estimate the essential matrix
        E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        # Estimate Rotation and translation vectors
        _, self.cur_R, self.cur_t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, self.K)
        # Triangulation, returns 3-D point cloud
        self.new_cloud = self.triangulatePoints(self.cur_R, self.cur_t)
        # The new frame becomes the previous frame
        self.frame_stage = STAGE_DEFAULT_FRAME
        self.px_ref = self.px_cur
        self.last_cloud = self.new_cloud


    def processFrame(self):
        # The images or roi used for the VO process (feature detection and tracking)
        prev_img, cur_img = self.last_frame, self.new_frame
        # De-rotate the images
        # prev_img = deRotateImage(prev_img)
        cur_img = deRotateImage(cur_img) 
        # Match features
        matches = self.matcher.knnMatch(self.px_ref, self.px_cur, k=2)
        # Filter matches to get good points
        self.px_ref, self.px_cur = filterMatches(matches, self.px_ref, self.px_cur)
        # Remove duplicates
        # self.px_ref = removeDuplicates(self.px_ref)
        # self.px_cur = removeDuplicates(self.px_cur)    
        # Refine matches
        F, _ = cv2.findFundamentalMat(self.px_ref, self.px_cur, cv2.FM_RANSAC)
        #
        self.px_ref, self.px_cur = betterMatches(F, self.px_ref, self.px_cur)
        # Draw matches 
        # drawMatchField(cur_img, self.px_ref, self.px_cur, F) 
        # Track features: Obtain feature correspondence points 
        self.px_ref, self.px_cur, px_diff = KLT_featureTracking(prev_img, cur_img, self.px_ref)
        # Verify if the current frame is going to be skipped
        self.skip_frame = self.frame_Skip(px_diff)

        if self.skip_frame:
            if self.px_ref.shape[0] < kMinNumFeature:
                self.px_cur = self.detectNewFeatures(prev_img)
                self.px_ref = self.px_cur
                self.last_cloud = self.new_cloud
            return

        try:
            E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            _, R, t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, self.K)
        except cv2.error as e:
            print(f"Error occurred during pose recovery: {e}")
            if self.px_ref.shape[0] < kMinNumFeature:
                self.px_cur = self.detectNewFeatures(cur_img)
                self.px_ref = self.px_cur
                self.last_cloud = self.new_cloud
            return

        self.new_cloud = self.triangulatePoints(R, t)
        self.Scale = self.getRelativeScale()

        # Check for NaN values in the variables
        if np.any(np.isnan(t)) or np.any(np.isnan(R)) or np.isnan(self.Scale):
            print("Warning: NaN values detected. Skipping frame.")
            if self.px_ref.shape[0] < kMinNumFeature:
                self.px_cur = self.detectNewFeatures(prev_img)
                self.px_ref = self.px_cur
                self.last_cloud = self.new_cloud
            return
        elif t[2] > t[0] and t[2] > t[1]:
            self.cur_t = self.cur_t + self.Scale * self.cur_R.dot(t) # t_f = t_f + scale*(R_f*t);
            self.cur_R = R.dot(self.cur_R) # R_f = R*R_f;

        if self.px_ref.shape[0] < kMinNumFeature:
            self.px_cur = self.detectNewFeatures(cur_img)

        self.px_ref = self.px_cur
        self.last_cloud = self.new_cloud



    def update(self, img):
        assert img.ndim == 2, "Frame: provided image is not grayscale"
        self.new_frame = img

        if self.frame_stage == STAGE_DEFAULT_FRAME:
            self.processFrame()
        elif self.frame_stage == STAGE_SECOND_FRAME:
            self.processSecondFrame()
        elif self.frame_stage == STAGE_FIRST_FRAME:
            self.processFirstFrame()

        if self.skip_frame:
            return False

        self.last_frame = self.new_frame
        return True









class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')

        image_raw_topic = '/kinect_camera/image_raw'
        depth_image_topic = '/kinect_camera/depth/image_raw'
        camera_info = '/kinect_camera/camera_info'

        self.bridge = CvBridge()

        self.prev_keypoint = None
        self.prev_image = None
        self.pp = None # principal point of the camera #  (image_width / 2, image_height / 2).
        self.focal_length = None
        self.trajectory = [] 
        self.traj = np.zeros((600,600,3), dtype=np.uint8)
        plt.figure()

        self.image_subscription = self.create_subscription(
            Image,
            image_raw_topic,
            self.image_callback,
            10)

        # self.depth_image_subscription = self.create_subscription(
        #     Image,
        #     depth_image_topic,
        #     self.depth_image_callback,
        #     10)

        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            camera_info,
            self.camera_info_callback,
            10)
    
    # -----------------------------------------
    # -----------------------------------------  

    def image_callback(self, msg):
        # cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8") # coloured image
        cv_image = self.bridge.imgmsg_to_cv2(msg, "mono8") # grayscale image
        if self.focal_length is None and self.pp is None:
            return

        # Create a CLAHE object (contrast limiting adaptive histogram equalization)
        clahe = cv2.createCLAHE(clipLimit=5.0)
        cv_image = clahe.apply(cv_image)
        
        self.vo.update(cv_image)

        cur_t = self.vo.cur_t
        x, y, z = cur_t[0], cur_t[1], cur_t[2]
        draw_x, draw_y = int(x)+290, int(z)+290

        cv2.circle(self.traj, (draw_x,draw_y), 1, (0,0,255), 2)
        cv2.rectangle(self.traj, (10, 20), (600, 60), (0,0,0), -1)
        text = "Coordinates: x=%2fm y=%2fm z=%2fm"%(x,y,z)
        cv2.putText(self.traj, text, (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)
        cv2.imshow('Road facing camera', cv_image)
        cv2.imshow('Trajectory', self.traj)
        cv2.waitKey(1)
        cv2.imwrite('map.png', self.traj)
    
        # if len(self.trajectory) >= 50:
        #     self.trajectory.pop(0)
        # self.trajectory.append((draw_x, draw_y))
        # self.plot_trajectory()
        # cv2.imshow("Image Window", cv_image)
        # cv2.waitKey(1)


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

    def depth_image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "32FC1")
        # cv2.imshow("Depth Image Window", cv_image)
        # cv2.waitKey(1)
    
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
        dist_coeff = msg.d  # Distortion coefficients
        # Image dimensions
        image_height = msg.height  # Image height
        image_width = msg.width  # Image width
        # self.get_logger().info('Camera Info: fx=%f, fy=%f, \
        #                        ppx=%f, ppy=%f, distortion_coeff=%s, \
        #                        image_height=%d, image_width=%d' \
        #                        % (fx, fy, ppx, ppy, distortion_coeff, image_height, image_width))
        if self.focal_length is None and self.pp is None:
            # width, height, fx, fy, cx, cy, k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0
            # cam = PinholeCamera(image_width, image_height, fx, fy, ppx, ppy)
            cam = np.array([[fx, 0.0, ppx],[0.0, fy, ppy],[0.0, 0.0, 1.0]], dtype=np.float32)
            self.vo = VisualOdometry(cam)     

        self.camera_matrix = np.array([[fx, 0, ppx], [0, fy, ppy], [0, 0, 1]])
        self.pp = (ppx, ppy) # (ppx + ppy) / 2 # principal point of the camera #  (image_width / 2, image_height / 2).
        self.focal_length = (fx + fy) / 2 


 
    # -----------------------------------------
    # -----------------------------------------  

def main(args=None):
    rclpy.init(args=args)

    image_subscriber = ImageSubscriber()

    rclpy.spin(image_subscriber)

    image_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()




















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
