#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np


# //Algorithm Outline:
# //Capture images: ItIt, It+1It+1,
# //Undistort the above images.
# //Use FAST algorithm to detect features in ItIt, and track those features to It+1It+1. A new detection is triggered if the number of features drop below a certain threshold.
# //Use Nister’s 5-point alogirthm with RANSAC to compute the essential matrix.
# //Estimate R,tR,t from the essential matrix that was computed in the previous step.
# //Take scale information from some external source (like a speedometer), and concatenate the translation vectors, and rotation matrices.


class MonoVisualOdometry:
    def __init__(self, ppx, ppy, fx, fy, cam_to_ground_height, frame_threshold, keypoint_threshold, detector_type, matcher_type):
        self.pp = (ppx + ppy) / 2 # principal point of the camera #  (image_width / 2, image_height / 2).
        self.focal_length = (fx + fy) / 2 
        self.frame_threshold = frame_threshold
        self.keypoint_threshold = keypoint_threshold
        self.cam_to_ground_height = cam_to_ground_height
        self.detector_type = detector_type
        self.matcher_type = matcher_type
        self.frame_count = 0
        self.R = np.eye(3)
        self.t = np.zeros((3, 1))

    def process_frame(self, img1, img2):
        # Detect features
        if self.detector_type == 'SIFT':
            detector = cv2.SIFT_create()
        elif self.detector_type == 'FAST':
            detector = cv2.FastFeatureDetector_create()
        else:
            raise ValueError('Invalid detector type')

        if self.frame_count % self.frame_threshold == 0 or len(self.kp1) < self.keypoint_threshold:
            kp1, des1 = detector.detectAndCompute(img1, None)
        else:
            kp1, des1 = self.kp1, self.des1
        kp2, des2 = detector.detectAndCompute(img2, None)

        # Match features
        if self.matcher_type == 'BF':
            matcher = cv2.BFMatcher()
            matches = matcher.knnMatch(des1, des2, k=2)
        elif self.matcher_type == 'LK':
            # Convert keypoints to points
            points1 = np.float32([kp.pt for kp in kp1])
            points2, st, _ = cv2.calcOpticalFlowPyrLK(img1, img2, points1, None)
            # Filter points based on status
            matches = [cv2.DMatch(i, i, 0) for i, m in enumerate(st) if m]
        else:
            raise ValueError('Invalid matcher type')

        # Apply ratio test for BF matcher
        if self.matcher_type == 'BF':
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append(m)
            matches = good

        # Estimate essential matrix
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        E, mask = cv2.findEssentialMat(src_pts, dst_pts, self.focal_length, self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)

        # Recover pose
        _, R, t, _ = cv2.recoverPose(E, src_pts, dst_pts, self.focal_length, self.pp)

        # Estimate scale
        scale = self.cam_to_ground_height / t[1]

        # Update pose
        self.t += scale * self.R.dot(t)
        self.R = R.dot(self.R)

        # Update frame count and keypoints
        self.frame_count += 1
        self.kp1, self.des1 = kp2, des2

        return self.R, self.t









