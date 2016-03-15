#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2

# local modules
from video import create_capture
from common import clock, draw_str

FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH    = 6
flann_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2

MIN_MATCH_COUNT = 10

class CarImage:
    def __init__(self, image, car_rect, plate_rect, hm):
        self.image = image
        self.car_rect = car_rect
        self.plate_rect = plate_rect
        self.himself_matches = hm

class Car:
    def __init__(self, id):
        self.id = id
        self.images = {};
        self._img_idx = 0
        self.detector = cv2.ORB_create( nfeatures = 1000 )
        self.matcher = cv2.FlannBasedMatcher(flann_params, {})
        self.last_similar = 0

    def add(self, image, car_rect, plate_rect):
        self._img_idx += 1
        # get number of matches beetween himself
        m = self.get_number_of_matches(image, image)
        self.images[self._img_idx] = CarImage(image, car_rect, plate_rect, m)

    def get(self):
        return self.images[self._img_idx]

    def get_image(self):
        return self.get().image

    def get_himself_matches(self):
        return self.get().himself_matches
        
    def get_car_rect(self):
        return self.get().car_rect
    
    def get_rects(self):
        car = self.get()
        return car.car_rect, car.plate_rect

    def get_number_of_matches(self, image, image2, preview=False):
        kp_car, des_car     = self.detector.detectAndCompute(image, None)
        kp_frame, des_frame = self.detector.detectAndCompute(image2, None)
        matches = self.matcher.knnMatch(des_car, des_frame, k = 2)
        matches = [ x for x in matches if len(x) == 2 ]
        matchesMask = [[0,0] for i in xrange(len(matches))]
        true_matches = 0

        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                matchesMask[i]=[1,0]
                true_matches += 1
                    
        draw_params = dict(matchColor = (0,255,0),
                           singlePointColor = (255,0,0),
                           matchesMask = matchesMask,
                           flags = 0)
        if preview:
            x = cv2.drawMatchesKnn(self.get_image(),kp_car,image,kp_frame,matches,None,**draw_params)
            cv2.imshow('x', x)
        return true_matches

    def is_similar(self, image, threshold=1, preview = False):
        # except:
        matches = self.get_number_of_matches(self.get_image(), image, preview)
        normalize = float(matches)/self.get_himself_matches()
        car = self.get()
        self.last_similar = normalize
        print("Car %d matches %d/%d, normalize: %f" % (self.id, matches, self.get_himself_matches(), normalize))
        is_similar = normalize >= threshold
        return is_similar

class Finder:
    def __init__(self, car_cascade, plate_cascade, scaleFactor=1.3, minNeighbors=10, minSize=(50,50)):
        self.car_cascade = cv2.CascadeClassifier(car_cascade)
        self.plate_cascade = cv2.CascadeClassifier(plate_cascade)
        self.cars = []
        self.visible_cars = []
        self._car_scaleFactor = scaleFactor
        self._car_minNeighbors = minNeighbors
        self._car_minSize = minSize
        self.image = None
        self._id = 0
        self._frame_no = 0

    def set_image(self, image):
        self.image = image

    def detect_cars(self, must_has_plate = True):
        if self.image == None:
            print("Warning! Image is not sets")
            return []

        cars_with_plate = []
        
        img = self.image.copy()
        cars_rects = self.car_cascade.detectMultiScale(img,
                                                  scaleFactor  = self._car_scaleFactor,
                                                  minNeighbors = self._car_minNeighbors,
                                                  minSize      = self._car_minSize,
                                                  flags        = cv2.CASCADE_SCALE_IMAGE)
        if len(cars_rects) == 0:
            return 
        cars_rects[:,2:] += cars_rects[:,:2]
        for car_rect in cars_rects:
            car_img = self.crop(car_rect, img, copy = True)
            plate_rect = self.plate_cascade.detectMultiScale(car_img,
                                                              scaleFactor  = self._car_scaleFactor,
                                                              minNeighbors = self._car_minNeighbors,
                                                              minSize      = (60,27),
                                                              flags        = cv2.CASCADE_SCALE_IMAGE)
            if len(plate_rect) == 1:
                plate_rect[:,2:] += plate_rect[:,:2]
                plate_rect = plate_rect[0]
                cars_with_plate.append([car_rect, plate_rect])

        if len(cars_with_plate) == 0:
            return []

        for (car_rect, plate_rect) in cars_with_plate:
            car_img = self.crop(car_rect, img, copy = False)
            maybe_exist = self.already_exist(car_img)
            if maybe_exist:
                maybe_exist.add(car_img, car_rect, plate_rect)
                self.visible_cars.append(maybe_exist)
            else:
                car = self.new_cars(car_img, car_rect, plate_rect)
                self.visible_cars.append(car)

    def visualize(self):
        vis_img = self.image.copy()
        self._frame_no += 1
        
        for car in self.visible_cars:
            car_rect, plate_rect = car.get_rects()
            str_pos = (car_rect[0], car_rect[1]-20)
            draw_str(vis_img, str_pos, 'car no %d: similar: %f' % (car.id, car.last_similar))
            self._draw_rect(vis_img, car_rect)
            # move plate_rect to car_rect coordinates
            plate_rect[0] = car_rect[0]+plate_rect[0]
            plate_rect[1] = car_rect[1]+plate_rect[1]
            plate_rect[2] = car_rect[0]+plate_rect[2]
            plate_rect[3] = car_rect[1]+plate_rect[3]

            self._draw_rect(vis_img, plate_rect)

        cv2.imshow('facedetect', vis_img)
        file_name = "/tmp/detect/%s.jpg" % str(self._frame_no).zfill(5)
        cv2.imwrite(file_name, vis_img)

    def crop(self, rect, img, copy = True):
        x1, y1, x2, y2 = rect
        car_img = img[y1:y2, x1:x2]
        if copy:
            return car_img.copy()
        else:
            return car_img

    def already_exist(self, car_img):
        for car in self.cars:
            if car.is_similar(car_img, threshold = 0.15, preview = False):
                return car
        return None

    def new_cars(self, car_img, car_rect, plate_rect):
        car = Car(self._id)
        self._id += 1
        car.add(car_img, car_rect, plate_rect)
        self.cars.append(car)
        return car

    def clear_visible(self):
        self.visible_cars = []

    def _draw_rect(self, img, rect, color=(255,0,0), thickness=2):
        x1, y1, x2, y2 = rect
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

if __name__ == '__main__':
    import sys, getopt
    print(__doc__)

    args, video_src = getopt.getopt(sys.argv[1:], '', ['car-cascade=', 'plate-cascade='])
    try:
        video_src = video_src[0]
    except:
        video_src = 0
    args = dict(args)
    car_fn = args.get('--car-cascade', "car.xml")
    plate_fn  = args.get('--plate-cascade', "plate.xml")

    cam = create_capture(video_src)
    finder = Finder(car_fn, plate_fn)

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # gray = cv2.equalizeHist(gray)
        finder.clear_visible()
        finder.set_image(gray)
        finder.detect_cars()
        finder.visualize()

        if 0xFF & cv2.waitKey(5) == 27:
            break
    cv2.destroyAllWindows()
