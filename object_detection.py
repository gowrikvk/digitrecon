# coding: utf-8
"""
 Computer Vision 2018 (week 3,4): Example of feature matching & homography
   Hiroaki Kawashima <kawashima@i.kyoto-u.ac.jp>
   https://github.com/hkawash/feature-matching-example
   Ref: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
"""

import numpy as np
import cv2
import imutils
#import deepcopy

class ObjectDetector:
    """ Realtime planar object detector using feature matching """

    def __init__(self):
        # Feature-point detector
        self.feature_detector = cv2.AKAZE_create() # Use AKAZE
        #self.detector = cv2.ORB_create() # Use ORB

        # VideoCapture setting
        # self.vidcap = cv2.VideoCapture(0)
        # self.vidcap.set(3, 640) # width
        # self.vidcap.set(4, 480) # height
        # self.vidcap.set(5, 15)  # frame rate

        # ROI (Region-Of-Interest) to register a target object
        self.sub_topleft = [100, 220] # [0, 0] # [y,x]
        self.sub_width = 200 #640
        self.sub_height = 200 #480
        self.sub_bottomright = [self.sub_topleft[0] + self.sub_height - 1,\
                                self.sub_topleft[1] + self.sub_width - 1]
        self.rect_color = (0, 255, 0) # green
        self.rect_thickness = 3
        self.rect_tl_outer_xy = (self.sub_topleft[1] - self.rect_thickness,\
                                 self.sub_topleft[0] - self.rect_thickness)
        self.rect_br_outer_xy = (self.sub_bottomright[1] + self.rect_thickness,\
                                 self.sub_bottomright[0] + self.rect_thickness)

        self.ratio = 0.75  # Threshold for the distance of feature (descriptor) vectors
        self.registered = False
        self.min_match_count = 4

    def register(self):
        """ Register target object """

        # print("\nHold a target object close to the camera.")
        # print("Make sure the object fully covers (background is not visible) inside the rectangle.")
        # print("Then, press 'r' to register the object.\n")

        temp = True
        #while self.vidcap.isOpened():
        if temp == True:
            #frame = self.vidcap.read()
            frame = cv2.imread('box.jpg',50) # trainI)
            #frame = cv2.imread('ryan2.jpg',50) # trainI)
            frame = cv2.imread('doll.jpg') # trainI)

            cv2.rectangle(frame, self.rect_tl_outer_xy, self.rect_br_outer_xy,\
                          self.rect_color, self.rect_thickness)
            cv2.imshow("Registration (press 'r' to register)", frame)
            cv2.waitKey(0)
            #subimg = frame[self.sub_topleft[0]:(self.sub_topleft[0] + self.sub_height),self.sub_topleft[1]:(self.sub_topleft[1] + self.sub_width)]
            # cv2.imshow("Registration sub image : ", subimg)
            # cv2.waitKey(0)
            self.kp0, self.des0 = self.feature_detector.detectAndCompute(frame, None)
            self.querying = frame
            self.registered = True

            #if cv2.waitKey(1) & 0xFF == ord('r'):
            #    subimg = frame[self.sub_topleft[0]:(self.sub_topleft[0] + self.sub_height),
            #                   self.sub_topleft[1]:(self.sub_topleft[1] + self.sub_width)]
                # self.kp0, self.des0 = self.feature_detector.detectAndCompute(subimg, None)
                # self.querying = subimg
                # self.registered = True
                # break

    def detect(self):
        """ Find object using feature points """

        # if not self.registered:
        #     print("Call 'register()' first.")
        #     return

        # print("Start detection...")
        # print("Press 'q' to quit.\n")

        bf = cv2.BFMatcher()  # Prepare a Blute-Force (BF) matcher

        temp = True
        #while self.vidcap.isOpened():
        if temp == True:
            print("Into matching process")
            frame = cv2.imread('box_in_scene.jpg') # trainI
            #frame = cv2.imread('ryan.jpg') # trainI
            frame = cv2.imread('chinchq.jpg') # trainI
            bkp_frame = cv2.imread('chinchq.jpg')
            #frame = cv2.imread('cblankcq.jpg')

            # Keypoint (kp) detection and calculate descriptors (des)
            kp, des = self.feature_detector.detectAndCompute(frame, None)
            winframe=imutils.resize(frame, 1000,1000)
            cv2.imshow(" Actual Img", winframe)
            cv2.waitKey(0)
            cv2.imshow(" Template Image", self.querying)
            cv2.waitKey(0)
            cv2.imshow(" no .... ", self.querying)
            cv2.waitKey(0)
            # Apply blute-force knn matching between keypoints
            #matches = bf.knnMatch(self.des0, des, k=2)
            matches = bf.knnMatch(self.des0, des, k=2)

            #print(matches)
            # Adopt only good feature matches
            good = [[m] for m, n in matches if m.distance < self.ratio * n.distance]
            #print(good)
            #print(self.min_match_count)

            # Find Homography
            if len(good) > self.min_match_count:
                src_pts = np.float32([self.kp0[m[0].queryIdx].pt for m in good]).reshape(-1, 1, 2)
                #print(src_pts)
                dst_pts = np.float32([kp[m[0].trainIdx].pt for m in good]).reshape(-1, 1, 2)
                #print(dst_pts)


                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                h, w, _ = self.querying.shape  # Assume color camera
                pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                #print(pts)
                dst = cv2.perspectiveTransform(pts, M)
                print("box coords" + str(dst))

                for i, coords in enumerate(dst):
                    #print(i, coords)
                    if i == 0:
                        x1, y1 = coords[0]
                    elif i == 1:
                        x2, y2 = coords[0]
                    elif i == 2:
                        x3, y3 = coords[0]
                    else:
                        x4, y4 = coords[0]

                #cropped_img = frame[508:675, 2936:3000]

                #cv2.imshow("Amount portion : ",cropped_img)
                #cv2.waitKey()
                # print("Y Axis")
                # print(y1, y2, y3, y4)
                # print("X Axis")
                # print(x1, x2, x3, x4)
                # print(int(x2)+100)
                cropped_img = bkp_frame[int(y1):int(y3), int(x2):]

                frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 2, cv2.LINE_AA)
                #print(frame)

            # Visualize the matches
            #draw_params = dict(flags=2)
            draw_params = dict(matchColor=(0, 255, 0), singlePointColor=(0, 0, 255), flags=0)
            img = cv2.drawMatchesKnn(self.querying, self.kp0, frame, kp, good, None, **draw_params)

            #cv2.imshow("Detection (press 'q' to quit)", img)

            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break

            #cv2.figure(figsize=IMAGE_SIZE)
            img=imutils.resize(img, 1000,2000)
            cv2.imshow("feature match : ",img)
            cv2.waitKey()

            cv2.imshow("Amount portion : ",cropped_img)
            cv2.waitKey()

    def close(self):
        """ Release VideoCapture and destroy windows """
        #self.vidcap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    obj_detector = ObjectDetector()
    obj_detector.register()
    obj_detector.detect()
    obj_detector.close()
