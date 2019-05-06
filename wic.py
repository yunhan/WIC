import cv2
import numpy as np
import math
import json
from math import atan2, degrees
from scipy.interpolate import splprep, splev


def draw_pts(pts):
    if pts is not None:
        print(pts)
        for cnt in pts:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)


def draw_box(box):
    if box is not None:
        # print('draw_box', box)
        for b in box:
            x, y, w, h = b
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)


def draw_line_pts(line):
    if line is not None:
        # print(line)
        i = 0
        while i < len(line)-1:
            cv2.line(img, (line[i][0]), (line[i+1][0]), (0, 0, 255))
            i = i+1


def draw_iqi(pts):
    if pts is not None:
        for pt in pts:
            cv2.line(img, (pt[0]), (pt[1]), (255, 255, 0))


def draw_focus_area(pts):
    if pts is not None:
        for pt in pts:
            cv2.circle(img, pt, 1, (0, 255, 255), 1)


def draw_mser_boxes(pts):
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in pts]
    cv2.polylines(img, hulls, 1, (0, 255, 0))
    for contour in hulls:
        cv2.drawContours(img, [contour], -1, (0, 255, 255), -1)


def filter_box(pts):
    filtered = []
    if pts is not None:
        for cnt in pts:
            x, y, w, h = cv2.boundingRect(cnt)
            if 20 < x < img_w-50 and 100 < y < img_h-50 and w*h > 100:
                # filter: boxes close to the border
                # filter: small dots
                filtered.append(cnt)
    return filtered


def get_intersection_ratio(p1, p2):
    p1_x, p1_y, p1_w, p1_h = cv2.boundingRect(p1)
    p2_x, p2_y, p2_w, p2_h = cv2.boundingRect(p2)

    x_left = max(p1_x, p2_x)
    y_top = max(p1_y, p2_y)
    x_right = min(p1_x + p1_w, p2_x + p2_w)
    y_bottom = min(p1_y + p1_h, p2_y + p2_h)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = p1_w * p1_h
    bb2_area = p2_w * p2_h

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    return iou


def merge_box(pts):
    merged = []
    overlapped = False
    if pts is not None:
        for i, p1 in enumerate(pts):
            if len(merged) == 0:
                merged.append(p1)
            overlapped = False
            for j, p2 in enumerate(merged):
                ratio = get_intersection_ratio(p1, p2)
                if ratio != 0:
                    overlapped = True
                    new_cnt = np.vstack((p1, p2))
                    hull = cv2.convexHull(new_cnt)
                    merged[j] = hull[:, 0, :]
            if  overlapped == False:
                merged.append(p1)
    return merged


def line_removed_filter_box(pts):
    filtered = []
    for cnt in pts:
        x, y, w, h = cv2.boundingRect(cnt)
        # print(x)
        # print(img_w)
        print('......')
        # aspect_ratio = float(w) / h
        # if aspect_ratio < 1.5:
        filtered.append(cnt)
    return filtered


# sort recs by x-axis
def sort_box(pts):
    sorted_box = []
    counts = 0
    if pts is not None:
        for pt in pts:
            cv2.boundingRect(pt)
            sorted_box.append(cv2.boundingRect(pt))
            counts = counts + 1
        sorted_box = sorted(sorted_box)
    return sorted_box


# distinguish left,label,right
def group_rec(recs):
    left, label, right = [], [], []
    if recs is not None:
        n = len(recs)
        left.append(recs[0])
        right.append(recs[n-1])

        if recs[1][0]-recs[0][0] < 10:
            left.append(recs[1])
        else:
            label.append(recs[1])

        i = 2
        while i < n-2:
            label.append(recs[i])
            i = i+1

        if recs[n-1][0]-recs[n-2][0] < 10:
            right.append(recs[n-2])
        else:
            label.append(recs[n-2])

    return left, label, right


# merge label rec
def merge_label_rec(label):
    x1 = img_w
    y1 = img_h
    x2 = 0
    y2 = 0
    if label is not None:
        for pt in label:
            if pt[0] < x1:
                x1 = pt[0]
            if pt[1] < y1:
                y1 = pt[1]
            if pt[0] + pt[2] > x2:
                x2 = pt[0] + pt[2]
            if pt[1] + pt[3] > y2:
                y2 = pt[1] + pt[3]
        label = [(x1 - 20, y1 - 5, x2 - x1 + 40, y2 - y1 + 10)]
    return label


def iqi(input_img, input_label):
    # crop iqi area
    if input_label is not None:
        input_label = input_label[0]
        label_left_x, label_left_y = input_label[0], input_label[1]
        label_right_x, label_right_y = input_label[0]+input_label[2], input_label[1]+input_label[3]
    else:
        label_left_x = int(img_w*0.2)
        label_left_y = int(img_h)
        label_right_x = int(img_w*0.8)
    input_img = input_img[50:label_left_y, label_left_x:label_right_x]  # crop img [y1:y2, x1:x2] for interested area
    input_gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    input_gray = cv2.equalizeHist(input_gray)
    input_gray = cv2.Canny(input_gray, 80, 300, apertureSize=3, L2gradient=True)
    lines = cv2.HoughLinesP(input_gray, 1, np.pi / 180, 50, maxLineGap=10)

    iqi_lines = []
    slope = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # get vertical lines
            if x2 - x1 != 0:
                slope = (y2 - y1) / (x2 - x1)
            if math.fabs(slope) > 0.5 and abs(y2 - y1) > 50:
                h = 50
                w = label_left_x
                iqi_lines.append([(x1+w, y1+h), (x2+w, y2+h)])
    return iqi_lines


def center_line(input_img, input_left, input_right):
    if input_left is not None:
        input_left = input_left[0]
        input_right = input_right[0]
        input_y = input_left[1]
        input_left_x = input_left[0]
        input_right_x = input_right[0]
    else:
        input_y = img_h
        input_left_x = img_w * 0.2
        input_right_x = img_w * 0.8

    input_img = input_img[50:input_y, input_left_x:input_right_x]  # crop img [y1:y2, x1:x2] for interested area

    input_gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    input_gray = cv2.equalizeHist(input_gray)
    _, thresh = cv2.threshold(input_gray, 200, 255, cv2.THRESH_TOZERO+cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    _, contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=cv2.contourArea)

    size = input_img.shape[0], input_img.shape[1]
    m = np.zeros(size, dtype=np.uint8)
    m = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(m, [cnt], 0, (255, 255, 255), -1)
    m_color = m.copy()
    m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    _, contours, hierarchy = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(m_color, contours, -1, (0, 255, 0), 1)

    width = m_color.shape[1]
    height = m_color.shape[0]

    center_line_pts = []
    upper_root_line_pts = []
    lower_root_line_pts = []
    for x in range(2, width-2):
        high = 0
        low = 140
        for y in range(2, height-2):
            if m_color[y][x][0] == 0 and m_color[y][x][1] == 255 and m_color[y][x][2] == 0:
                if y > high:
                    high = y
                if y < low:
                    low = y

        middle = int((high+low)/2)
        center_line_pts.append([(x+input_left_x, middle+50)])
        upper_root_line_pts.append([(x+input_left_x, high+50)])
        lower_root_line_pts.append([(x+input_left_x, low+50)])
    return center_line_pts, upper_root_line_pts, lower_root_line_pts


def get_focus_area(left, right, c_line):
    if c_line is not None and left is not None and right is not None:
        up_left_x = c_line[0][0][0] + c_line[0][0][0] - left[0][0]
        up_left_y = c_line[0][0][1] + c_line[0][0][1] - left[0][1]
        up_left = (up_left_x, up_left_y)
        down_left = (left[0][0], left[0][1])
        up_right_x = c_line[len(c_line[0])-2][0][0] + c_line[len(c_line[0])-2][0][0] - right[0][0]
        up_right_y = c_line[len(c_line[0])-2][0][1] + c_line[len(c_line[0])-2][0][1] - right[0][1]
        up_right = (up_right_x, up_right_y)
        down_right = (right[0][0], right[0][1])
    else:
        up_left = (0, 0)
        down_left = (0, img_h)
        up_right = (img_w, 0)
        down_right = (img_w, img_h)

    corner_points = []
    corner_points.append(up_left)
    corner_points.append(down_left)
    corner_points.append(up_right)
    corner_points.append(down_right)
    return corner_points


def remove_label_line(input_img, input_label):
    if input_label is not None:
        input_label = input_label[0]
        input_img = input_img[input_label[1]:input_label[3]+input_label[1], input_label[0]:input_label[2]+input_label[0]]
    else:
        input_img = img
    gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(gray, low_threshold, high_threshold)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 100  # angular resolution in radians of the Hough grid
    threshold = 60  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 10  # minimum number of pixels making up a line
    max_line_gap = 50  # maximum gap in pixels between connectable line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

    # Count the angle of the line and rotate the image to make the line horizontal
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                angle = degrees(atan2(y2 - y1, x2 - x1))
        rows, cols, _ = input_img.shape
        rotate = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        input_img = cv2.warpAffine(input_img, rotate, input_img.shape[1::-1], flags=cv2.INTER_LINEAR)

        linek = np.zeros((15, 21), dtype=np.uint8)
        linek[7, ...] = 1
        x = cv2.morphologyEx(input_img, cv2.MORPH_OPEN, linek, iterations=1)
        input_img -= x
    return input_img


def split_char(input_img):
    org = input_img.copy()
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    mser2 = cv2.MSER_create(_min_area=10, _max_area=180)
    img_points, img_bbox = mser2.detectRegions(input_img)
    img_points = merge_box(img_points)

    counts = 0
    for pt in img_points:
        x1, y1, w1, h1 = cv2.boundingRect(pt)
        # cv2.rectangle(org, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 1)
        single_char_img = org[y1:y1 + h1, x1:x1 + w1]
        filename = str(counts) + '.png'
        cv2.imwrite(filename, single_char_img)
        counts = counts + 1


img = cv2.imread(r'../Test_Images/25.png')
img = cv2.resize(img, (726, 257))
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_h = img.shape[0]
img_w = img.shape[1]

mser = cv2.MSER_create(_min_area=30, _max_area=800)
points, bbox = mser.detectRegions(img_gray)
# draw_mser_boxes(points)

points = filter_box(points)
points = merge_box(points)
rectangles = sort_box(points)
left_rec, label_rec, right_rec = group_rec(rectangles)
label_rec = merge_label_rec(label_rec)
# print(left_rec, label_rec, right_rec)

iqi_line = iqi(img, label_rec)
center_line, upper_root_line, lower_root_line = center_line(img, left_rec, right_rec)
focus_area_pt = get_focus_area(left_rec, right_rec, center_line)
draw_focus_area(focus_area_pt)

img_line_removed = remove_label_line(img, label_rec)
# split_char(img_line_removed)

# use for debugging
# draw_box(rectangles)
draw_box(label_rec)
draw_box(left_rec)
draw_box(right_rec)
draw_line_pts(center_line)
draw_line_pts(upper_root_line)
draw_line_pts(lower_root_line)
draw_iqi(iqi_line)
cv2.imshow('line_removed_label', img_line_removed)
cv2.imshow('draw_result', img)

cv2.waitKey(0)
cv2.destroyAllWindows()










