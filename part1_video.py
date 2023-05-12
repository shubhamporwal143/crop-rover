import cv2
import numpy as np
import matplotlib.pylab as plt
import math


def detect_edges(frame):
    #BGR to HSV transformation
    # filter for blue lane lines
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #cv2.imshow("hsv",hsv)
    #specifying a range of the color Blue.
    lower_b = np.array([19, 101, 0])
    upper_b = np.array([165, 255, 255])
    mask = cv2.inRange(hsv, lower_b, upper_b)
    #cv2.imshow("mask",mask)
    median = cv2.medianBlur(mask,15)
    th2 = cv2.adaptiveThreshold(median,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 5,1)
    #Detecting Edges of Lane Lines
    edges = cv2.Canny(th2, 100, 200)
    return edges


#region_of_interest
def region_of_interest(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)

    # only focus bottom half of the screen
    polygon = np.array([[
        (0, height),
        (0, height/2),
        (width, height/2),
        (width, height),
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)
    return cropped_edges

#detect line segment
def detect_line_segments(cropped_edges):
    # tuning min_threshold, minLineLength, maxLineGap is a trial and error process by hand
    rho = 1  # distance precision in pixel, i.e. 1 pixel
    angle = np.pi / 180  # angular precision in radian, i.e. 1 degree
    min_threshold = 46  # minimal of votes
    line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold, 
                                    np.array([]), minLineLength=5, maxLineGap=5)

    return line_segments

#function call for line segment


#combine linesegment and draw line on image 
def average_slope_intercept(frame, line_segments):
    lane_lines = []
    if line_segments is None:
        return lane_lines

    height, width, _ = frame.shape
    left_fit = []
    right_fit = []

    boundary = 1/3
    left_region_boundary = width * (1 - boundary)
    right_region_boundary = width * boundary

    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    if len(left_fit) > 0:
        lane_lines.append(make_points(frame, left_fit_average))

    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append(make_points(frame, right_fit_average))

    return lane_lines

def make_points(frame, line):
    height, width, _ = frame.shape
    slope, intercept = line
    y1 = height
    y2 = int(y1 * 1 / 2) 

    # bound the coordinates within the frame
    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    return [[x1, y1, x2, y2]]





def detect_lane(frame):
    
    edges = detect_edges(frame)
    cropped_edges = region_of_interest(edges)
    line_segments = detect_line_segments(cropped_edges)
    lane_lines = average_slope_intercept(frame, line_segments)
    
    return lane_lines


def display_lines(frame, lines, line_color=(0, 255, 0), line_width=6):
    line_image = np.zeros_like(frame)
    height,width,_ =frame.shape
    hr_st=((width//2)-width//10,height//2)
    hr_end=((width//2)+width//10,height//2)
    vr_st=(width//2,(height//2+height//10))
    vr_end=(width//2,(height//2-height//10))
    half_vr_line=height/2
    cv2.line(frame, hr_st, hr_end, (0, 255,255), 3)
    cv2.line(frame, vr_st,vr_end , (0, 255, 255), 3)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(frame, (x1, y1), (x2, y2), line_color, line_width)
    return frame,half_vr_line
def crosshair(frame,line_color=(0, 255, 255),line_width=2):
    height,width,_ =frame.shape
    hr_st=((width//2)-80,height//2)
    hr_end=((width//2)+80,height//2)
    vr_st=(width//2,(height//2+80))
    vr_end=(width//2,(height//2-80))
    center_point=(width//2,height//2)
    cv2.line(frame, hr_st, hr_end, (0, 255,255), 5)
    cv2.line(frame, vr_st,vr_end , (0, 255, 255), 5)
    return frame,center_point
def center_line(frame, lane_lines):
    height, width, _ = frame.shape
    _, _, left_x2, _ = lane_lines[0][0]
    _, _, right_x2, _ = lane_lines[1][0]
    mid = int(width/2)
    x_offset = int((left_x2 + right_x2) / 2)
    y_offset = int(height / 2)
    line_end=(x_offset,y_offset)
    cv2.line(frame,(x_offset,y_offset),(mid,height),(255,0,0),3)
    return frame,line_end

def distance_lr(center,line_center):
    x1,y1=center
    x2,y2=line_center
    distance=int(((x2-x1)**2 + (y2-y1)**2)**0.5)
    l = ["1"]
    #l=[distance]
    if distance==0:
        l.append("Straight")
    elif (x2-x1)>0:
        l.append("right")
    else:
        l.append("left")

    angle=math.tan(distance/lane_lines_image[1])
    #l.append(lane_lines_image[1])
    angles=math.degrees(angle)
    l.append(angles)
    return l

#cap = cv2.VideoCapture("video2.mp4")

cap = cv2.VideoCapture("v2_test.mp4")
while True:
    _,frame = cap.read()
    #cv2.imshow("frame",frame)
    if cv2.waitKey(10) == ord("q"):
        break
    else:
        try:
            edges = detect_edges(frame)
            cropped_edges = region_of_interest(edges)
            line_segments = detect_line_segments(cropped_edges)
            lane_lines = average_slope_intercept(frame,line_segments)
            lane_lines_image = display_lines(frame, lane_lines)
            center_line_image = center_line(frame,lane_lines)
            cross_hair=(crosshair(frame))
            dist=distance_lr(cross_hair[1],center_line_image[1])
            cv2.imshow("edges",edges)
            
            #cv2.imshow("center_line",center_line_image[0])
            
            
            #cv2.imshow("lane lines", lane_lines_image[0])
            
            cv2.imshow("crosshair",cross_hair[0])
            #cv2.imshow("crosshair",cv2.resize(cross_hair[0],(640,480)))
            
            print(dist)
        except:
            print(0)
cv2.destroyAllWindows()