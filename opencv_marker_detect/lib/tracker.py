"""Marker detection with OpenCV"""

import cv2
import numpy as np

SQUARE_PX = 60
WIDTH = SQUARE_PX * 5
HEIGHT = SQUARE_PX * 5

VALID_MARKERS = {
    1: [[1, 0, 1], [0, 0, 0], [0, 0, 1]],
    2: [[1, 0, 1], [0, 0, 1], [0, 0, 1]],
    3: [[1, 0, 1], [0, 0, 0], [0, 1, 1]],
    4: [[1, 1, 1], [0, 0, 0], [0, 0, 1]],
    5: [[1, 1, 1], [0, 0, 1], [0, 0, 1]],
    6: [[1, 1, 1], [0, 0, 0], [0, 1, 1]]
}



class Marker:
    """Marker class which gets designated to all valid markers"""

    def __init__(self, marker_id, contour, polygon, rotations=0):
        self.id = marker_id
        self.contour = contour
        self.polygon = polygon
        self.rotations = rotations

        self.cx, self.cy = self.position
        self.x, self.y = self.corners

    def __str__(self):
        return str(self.id)

    @property
    def position(self):
        m = cv2.moments(self.contour)
        x = int(m['m10']/m['m00'])
        y = int(m['m01']/m['m00'])
        return x, y

    @property
    def corners(self):
        x, y = map(np.squeeze, np.hsplit(np.squeeze(self.polygon), 2))
        return x, y

    @property
    def major_axis(self):
        r = self.rotations
        f = lambda z: z[(4-r) % 4] + int((z[(5-r) % 4] - z[(4-r) % 4]) / 2)
        x, y = map(f, [self.x, self.y])
        return x, y

    @property
    def minor_axis(self):
        r = self.rotations
        f = lambda z: z[(5-r) % 4] + int((z[(6-r) % 4] - z[(5-r) % 4]) / 2)
        x, y = map(f, [self.x, self.y])
        return x, y

    @property
    def area_vec(self):
        raise NotImplementedError

    def angle_to_point(self, pt):
        maj, pos, pt = map(np.array, [self.major_axis, self.position, pt])

        phi = np.arctan2(*(maj - pos))
        rho = np.arctan2(*(pt - pos))

        f = lambda x: x + 2*np.pi if x < 0 else x
        phi, rho = map(f, [phi, rho])

        return round(np.degrees(rho - phi))


def small_area(region):
    return cv2.contourArea(region) < 1e2


def not_quadrilateral(points):
    return len(points) != 4


def no_black_border(region):
    r1mean = np.mean(region[0:60, 0:300])
    r2mean = np.mean(region[240:300, 0:300])
    r3mean = np.mean(region[60:240, 0:60])
    r4mean = np.mean(region[60:240, 240:300])
    mean = r1mean + r2mean + r3mean + r4mean
    #mean = np.sum(map(np.mean, [region[0:60, 0:300],
    #                            region[240:300, 0:300],
    #                            region[60:240, 0:60],
    #                            region[60:240, 240:300]]))
    return mean > 50


def oriented_clockwise(polygon):
    x, y = map(np.squeeze, np.hsplit(np.squeeze(polygon), 2))
    cross = (x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0])
    return cross > 0


def transform_matrix(polygon):
    if oriented_clockwise(polygon):
        return np.float32([[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]])
    else:
        return np.float32([[0, 0], [0, HEIGHT], [WIDTH, HEIGHT], [WIDTH, 0]])


def parse_marker(marker):
    marker_data = np.zeros(shape=(3, 3), dtype=np.int)

    squares = ((x, y, i, j)
               for i, x in enumerate(range(60, 181, 60))
               for j, y in enumerate(range(60, 181, 60)))

    for x, y, i, j in squares:
        if np.mean(marker[x:x+60, y:y+60]) > 200:
            marker_data[i, j] = 1

    return marker_data


def validate_marker(marker):

    valid_markers = ((marker_id, valid_marker, rotations)
                     for marker_id, valid_marker in VALID_MARKERS.items()
                     for rotations in range(4))
    print(marker)
    for marker_id, valid_marker, rotations in valid_markers:
        if np.array_equal(marker, np.rot90(valid_marker, rotations)):
            return True, marker_id, rotations

    return False, None, None


def find_markers(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #blur = cv2.medianBlur(gray, 5)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    __, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)
    __, contours, __ = cv2.findContours(thresh.copy(), cv2.RETR_TREE,
                                    cv2.CHAIN_APPROX_SIMPLE)

    markers = dict()

    for contour in contours:
        if small_area(contour):
            continue

        eps = 0.05 * cv2.arcLength(contour, closed=True)
        polygon = cv2.approxPolyDP(contour, eps, closed=True)

        if not_quadrilateral(polygon):
            continue

        polygon_fl = np.float32(polygon)
        tr_matrix = transform_matrix(polygon)
        transform = cv2.getPerspectiveTransform(polygon_fl, tr_matrix)

        # FIXME: Choose algorithm
        # Reuse thresholded image,
        # -- OR --
        # Run warpPerspective on "gray" threshold the result again.
        # In the latter case, the call to thresh.copy() is redundant.
        sq_marker = cv2.warpPerspective(thresh, transform, (WIDTH, HEIGHT))
        #__, sq_marker_bin = cv2.threshold(sq_marker, 0, 255,
        #                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)


        if no_black_border(sq_marker):
            continue

        marker = parse_marker(sq_marker)
        valid_marker, marker_id, rotations = validate_marker(marker)

        if not valid_marker:
            continue

        markers[marker_id] = Marker(marker_id, contour, polygon, rotations)

    return markers