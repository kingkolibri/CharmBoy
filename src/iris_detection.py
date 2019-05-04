# Numpy is needed because OpenCV images in python are actually numpy arrays.
import numpy
import cv2


# https://gist.github.com/esmitt/61edc8ed6ccbc7a7e857074299449990

class IrisDetection():
    def __init__(self, image_path):
        '''
        initialize the class and set the class attributes
        '''
        self._img = None
        self._img_path = image_path
        self._pupil = None

    def load_image(self):
        """
        load the image based on the path passed to the class
        it should use the method cv2.imread to load the image
        it should also detect if the file exists
        """
        self._img = cv2.imread(self._img_path)
        # If the image doesn't exists or is not valid then imread returns None
        NoneType = type(None)
        if type(self._img) == NoneType:
            return False
        else:
            return True

    def detect_pupil(self):
        """
        This method should use cv2.findContours and cv2.HoughCircles() function from cv2 library to find the pupil
        and then set the coordinates for pupil circle coordinates
        """
        # Blur and change to gray scale
        blurimg = cv2.GaussianBlur(self._img, (5, 5), 0)

        grayimg = cv2.cvtColor(blurimg, cv2.COLOR_BGR2GRAY)

        # First binarize the image so that findContours can work correctly.
        # _, thresh = cv2.threshold(self._img, 100, 255, cv2.THRESH_BINARY)
        # Now find the contours and then find the pupil in the contours.
        # contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        circles = cv2.HoughCircles(blurimg, cv2.HOUGH_GRADIENT, 2, grayimg.shape[0] / 2)
        # Then mask the pupil from the image and store it's coordinates.

        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            # draw the outer circle
            cv.circle(cimg, center, radius, (0, 255, 0), 2)
            # draw the center of the circle
            cv.circle(cimg, center, 2, (0, 0, 255), 3)
            self._pupil = (center[0], center[1], radius)
        # for l in c:
        # OpenCV returns the circles as a list of lists of circles
        # for circle in l:
        # center = (circle[0], circle[1])
        # radius = circle[2]
        # cv2.circle(self._img, center, radius, (0, 0, 0), thickness=-1)

    def detect_iris(self):
        '''
        This method should use the background subtraction technique to isolate the iris from the original image
        It should use the coordinates from the detect_pupil to get a larger circle using cv2.HoughCircles()
        '''
        _, t = cv2.threshold(self._img, 195, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Find the iris using the radius of the pupil as input.
        c = cv2.HoughCircles(contours, cv2.HOUGH_GRADIENT, 2, self._pupil[2] * 2, param2=150)

        for l in c:
            for circle in l:
                center = (self._pupil[0], self._pupil[1])
                radius = circle[2]
                # This creates a black image and draws an iris-sized white circle in it.
                mask = numpy.zeros((self._img.shape[0], self._img.shape[1], 1), numpy.uint8)
                cv2.circle(mask, center, radius, (255, 255, 255), thickness=-1)
                # Mask the iris and crop everything outside of its radius.
                self._img = cv2.bitwise_and(self._img, mask)

    def start_detection(self):
        '''
        This is the main method that will be called to detect the iris
        it will call all the previous methods in the following order:
        load_image
        convert_to_gray_scale
        detect_pupil
        detect_iris
        then it should display the resulting image with the iris only
        using the method cv2.imshow
        '''
        if (self.load_image()):
            self.detect_pupil()
            self.detect_iris()
            cv2.imshow("Result", self._img)
            cv2.waitKey(0)
        else:
            print('Image file "' + self._img_path + '" could not be loaded.')


if __name__ == '__main__':
    id = iris_detection('../temp/webcam/1.jpg')
    id.start_detection()
