import numpy as np
import cv2
import os
import glob
from tqdm import tqdm
import shutil
import yaml
import argparse


class Calibration:
    def __init__(self, args):
        config_path = args.config_path
        self.chess_path = args.chess_path
        self.extrinsic_chess_path = args.extrinsic_chess_path

        self.param_path = args.param_path
        with open(config_path, "r", encoding='utf-8') as f:
            data = yaml.load(f.read(), Loader=yaml.FullLoader)
        self.mode =  data["calibration"]["mode"]
        self.image_size = (data["calibration"]["image_width"], data["calibration"]["image_height"])
        self.corner_height = data["calibration"]["corner_height"]
        self.corner_width = data["calibration"]["corner_width"]
        self.corner_radius = data["calibration"]["corner_radius"]

        self.rvecs, self.tvecs = None, None



    def cal_real_corner(self):
        obj_corner = np.zeros([self.corner_height * self.corner_width, 3], np.float32)
        if self.mode == 'normal':
            obj_corner[:, :2] = np.mgrid[0:self.corner_height, 0:self.corner_width].T.reshape(-1, 2)  # (w*h)*2
        elif self.mode == 'fisheye':
            obj_corner = np.zeros((1,self.corner_height * self.corner_width, 3), np.float32)
            obj_corner[0,:, :2] = np.mgrid[:self.corner_height, :self.corner_width].T.reshape(-1, 2)
        return obj_corner
    
    def calibrate(self):
        if os.path.exists(self.param_path):
            self.load_param()
            self.calibrate_extrinsic()
        else:
            self.calibrate_intrinsic()
            self.calibrate_extrinsic()
        self.save_param()
        


    def calibrate_intrinsic(self):
        filenames = glob.glob(self.chess_path+"/*.JPG") + glob.glob(self.chess_path+"/*.jpg") + glob.glob(self.chess_path+"/*.png")
        objs_corner = []
        imgs_corner = []
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        obj_corner = self.cal_real_corner()        
        for filename in filenames:
            chess_image = cv2.imread(filename)
            gray = cv2.cvtColor(chess_image, cv2.COLOR_BGR2GRAY)
            # find chessboard corners
            ret, img_corners = cv2.findChessboardCorners(gray, (self.corner_height, self.corner_width))
            # append to img_corners
            if ret:
                objs_corner.append(obj_corner)
                img_corners = cv2.cornerSubPix(gray, img_corners, winSize=(self.corner_radius//2, self.corner_radius//2),
                                              zeroZone=(-1, -1), criteria=criteria)
                imgs_corner.append(img_corners)
            else:
                print("Fail to find corners in {}.".format(filename))
        # calibration
        if self.mode == "fisheye":
            ret, self.matrix, self.dist, rvecs, tveces = cv2.fisheye.calibrate(objs_corner, imgs_corner, self.image_size, None, None, criteria=criteria)
        else:
            ret, self.matrix, self.dist, rvecs, tveces = cv2.calibrateCamera(objs_corner, imgs_corner, self.image_size, None, None)

        self.new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(self.matrix, self.dist, self.image_size, alpha=1)
        self.roi = np.array(roi)
        print('内参矩阵',self.matrix)
        print('最优内参矩阵',self.new_camera_matrix)
        print('畸变系数',self.dist)
        return ret

    def calibrate_extrinsic(self):
        filenames = glob.glob(self.extrinsic_chess_path+"/*.JPG") + glob.glob(self.extrinsic_chess_path+"/*.jpg") + glob.glob(self.extrinsic_chess_path+"/*.png")
        if (len(filenames)):
            filename = filenames[0]
            chess_image = cv2.imread(filename)
            gray = cv2.cvtColor(chess_image, cv2.COLOR_BGR2GRAY)
            undistort_gray = cv2.undistort(gray, self.matrix, self.dist, None, self.new_camera_matrix)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            obj_corner = self.cal_real_corner()
            ret, img_corners = cv2.findChessboardCorners(undistort_gray, (self.corner_height, self.corner_width))
            # append to img_corners
            if ret:
                img_corners = cv2.cornerSubPix(gray, img_corners, winSize=(self.corner_radius//2, self.corner_radius//2),
                                              zeroZone=(-1, -1), criteria=criteria)
            else:
                print("Fail to find corners in {}.".format(filename))
                return False
            
            ret, self.rvecs, self.tvecs, mis= cv2.solvePnPRansac(obj_corner, img_corners, self.matrix, self.dist)
            print('旋转向量', self.rvecs)
            print('平移向量', self.tvecs)

            
        else:
            return False


    def load_param(self):
        with open(self.param_path, "r", encoding='utf-8') as f:
            self.param = yaml.load(f.read(), Loader=yaml.FullLoader)
            self.matrix = np.array(self.param["matrix"])
            self.new_camera_matrix = np.array(self.param["new_matrix"])
            self.dist = np.array(self.param["dist"])

            self.rvecs = np.array(self.param["rvecs"])
            self.tvecs = np.array(self.param["tvecs"])


    def save_param(self):
        self.param = {
            "matrix": self.matrix.tolist(),
            "new_matrix": self.new_camera_matrix.tolist(),
            "dist": self.dist.tolist()
        }
        if self.rvecs is not None:
            self.param['rvecs'] = self.rvecs.tolist()
        if self.tvecs is not None:
            self.param['tvecs'] = self.tvecs.tolist()


        with open(self.param_path, "w", encoding='utf-8') as f:
            yaml.dump(self.param, f)



    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='app_config.yaml', help='config file path')
    parser.add_argument('--param_path', type=str, default='param.yaml', help='param file path')
    parser.add_argument('--chess_path', type=str, default='./chess', help='config file path')
    parser.add_argument('--extrinsic_chess_path', type=str, default='./extrinsic_chess', help='config file path')
    args = parser.parse_args()
    calibrator = Calibration(args)
    calibrator.calibrate()
