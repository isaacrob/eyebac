import cv2
import numpy as np
from functools import reduce
import time
from collections import deque
import sys
import progressbar
import ast
import json

FILENAME = '/home/ubuntu/IMG_0923.mov'
EYE_CASCADE_LOCATION = '/home/ubuntu/.local/lib/python3.5/site-packages/cv2/data/haarcascade_eye.xml'
FPS = 30
eye_finder = cv2.CascadeClassifier(EYE_CASCADE_LOCATION)
params = cv2.SimpleBlobDetector_Params()
params.filterByColor = True
blob_finder = cv2.SimpleBlobDetector_create(params)

def get_dilation_rates_from_video(video = FILENAME, scale_for_eye_detection = 10, show = False, save_to = '/home/ubuntu/result', stamps = None):
    cap = cv2.VideoCapture(video)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    buffer_len = 15
    ratio_buffer = deque([], maxlen = buffer_len)
    rate_buffer = deque([], maxlen = buffer_len)
    all_ratios = []
    all_rates = []
    all_averaged_rates = []

    i = 0

    with progressbar.ProgressBar(max_value = frame_count) as bar:
        while cap.isOpened():
            ret, frame_color = cap.read()
            if not ret:
                break
            # iris_frame = frame_color.copy()
            big_frame = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)
            # print(big_frame.shape)
            x, y = big_frame.shape
            frame = cv2.resize(big_frame, (y // scale_for_eye_detection, x // scale_for_eye_detection))
            # print(frame.shape)
            eyes = get_eye(frame)

            for eye_location in eyes:
                ex, ey, ew, eh = eye_location
                subframe = big_frame[scale_for_eye_detection*ey:(ey+eh)*scale_for_eye_detection, scale_for_eye_detection*ex:(ex+eh)*scale_for_eye_detection]
                pupil = get_pupil_from_eye(subframe)
                color_subframe = frame_color[scale_for_eye_detection*ey:(ey+eh)*scale_for_eye_detection, scale_for_eye_detection*ex:(ex+eh)*scale_for_eye_detection].copy()
                iris = get_iris_by_scaling_pupil(color_subframe, pupil)
                # print(pupil)
                px, py, pr = pupil
                ix, iy, ir = iris
                spx, spy, spr = map(lambda x: x // scale_for_eye_detection, pupil)
                bex = ex*scale_for_eye_detection
                bey = ey*scale_for_eye_detection
                bew = ew*scale_for_eye_detection
                beh = eh*scale_for_eye_detection
                if show:
                    cv2.circle(frame_color, (bex+px, bey+py), pr, (250, 0, 0), 1)
                    cv2.circle(frame_color, (bex+ix, bey+iy), ir, (0, 0, 250), 1)
                # cv2.rectangle(frame_color, (bex, bey), (bex+bew, bey+beh), (0, 250, 0), 3)

                ratio = ir / pr
                ratio = 10*ratio**3
                ratio_buffer.append(ratio)
                all_ratios.append(ratio)
                # ratio = int(10*ratio**3)
                # print(ratios)
                # reported_ratio = int(sum(this_ratio*9*.1**(buffer_len-i) for i, this_ratio in enumerate(ratios_buffer)))
                if show:
                    cv2.line(frame_color, (10, 10), (10, 10 + int(ratio)), (0, 250, 0), 3)

                rate_of_change = int(20 * (ratio_buffer[-1] - ratio_buffer[0]) / buffer_len)
                rate_buffer.append(rate_of_change)
                reported_rate = int(np.abs(sum(this_rate*.5**(buffer_len-i) for i, this_rate in enumerate(rate_buffer))))
                if show:
                    cv2.line(frame_color, (20, 10), (20, 10 + reported_rate), (250, 250, 0), 3)

                all_rates.append(rate_of_change)
                all_averaged_rates.append(reported_rate)

                # do polar warp on iris and then stretch it
                # then eigen-based recognition
                # top_corner_x = iy-ir
                # top_corner_y = ix-ir
                # iris_frame = color_subframe[top_corner_x:top_corner_x+2*ir, top_corner_y:top_corner_y+2*ir, :].copy()
                # if 0 in iris_frame.shape:
                #     continue
                # h, w = iris_frame.shape[:2]
                # # y_mask, x_mask = np.ogrid[-iy:h-iy, -ix:w-ix]
                # y_mask, x_mask = np.ogrid[-ir:ir, -ir:ir]
                # mask_small = x_mask*x_mask + y_mask*y_mask <= pr**2
                # mask_big = x_mask*x_mask + y_mask*y_mask >= ir**2
                # mask = reduce(np.logical_or, [mask_small, mask_big])
                # # print(mask.shape)
                # # print(iris_frame.shape)
                # iris_frame[mask] = 0
                # if show:
                #     cv2.imshow('iris frame', iris_frame)
                #     cv2.waitKey(1)

                # # now do polar warp
                # warped_iris = cv2.warpPolar(iris_frame, dsize = iris_frame.shape[:2], center = (ir, ir), maxRadius = ir, flags = cv2.WARP_POLAR_LINEAR)
                # warped_iris = cv2.resize(warped_iris[:, pr*2:, :], warped_iris.shape[:2])
                # if show:
                #     cv2.imshow("warped iris", warped_iris)
                # 
                # # now for eigen-based recognition and PCA analysis
                # # TODO

            # cv2.imshow('frame', frame)
            if show:
                cv2.imshow('color frame', frame_color)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            i += 1
            bar.update(i)

    result = {'first_derivative': all_averaged_rates, 'dilation': all_ratios}
    if stamps is not None:
        breaking_point = int(find_breaking_point(stamps, all_averaged_rates))
        result['breaking_point'] = breaking_point
    result = json.dumps(result)
    with open(save_to, 'w+') as file_save_to:
        file_save_to.write(result)

    return (all_ratios, all_averaged_rates)

def get_eye(frame):
    eyes = eye_finder.detectMultiScale(frame, minNeighbors = 10, scaleFactor = 1.1)
    # print(eyes)
    return eyes

def get_iris_by_scaling_pupil(frame, pupil, radius_ratio = 1.1, mask_thickness = 2, scaling_factor = 1.03, color_noise_dist = 20):
    x, y, r = pupil
    r *= radius_ratio
    h, w, colors = frame.shape
    y_mask, x_mask = np.ogrid[-y:w-y, -x:h-x]
    mask_small = x_mask*x_mask + y_mask*y_mask <= (r+mask_thickness)**2 
    mask_big = x_mask*x_mask + y_mask*y_mask >= r**2
    mask = reduce(np.logical_and, [mask_small, mask_big])

    color_hists = []

    for i in range(colors):
        colored_frame = frame[:, :, i]
        average = np.median(colored_frame[mask])
        colored_frame = colored_frame.copy()
        colored_frame[colored_frame > average + color_noise_dist] = 255
        colored_frame[colored_frame < average - color_noise_dist] = 255
        colored_frame[colored_frame != 255] = 0
        color_hists.append(colored_frame)
    iris_frame = np.max(color_hists, 0)

    # cv2.imshow('iris_frame', iris_frame)

    while np.mean(iris_frame[y:, :][mask[y:, :]]) < 250:
        r *= scaling_factor
        mask_small = x_mask*x_mask + y_mask*y_mask <= (r+mask_thickness)**2 
        mask_big = x_mask*x_mask + y_mask*y_mask >= r**2
        mask = reduce(np.logical_and, [mask_small, mask_big])

    r = int(r / scaling_factor**4)
    # print("iris found at %d, %d, radius %d" % (x, y, r))

    return (x, y, r)

def get_iris_from_eye_and_pupil(frame, pupil, radius_ratio = 1.2, mask_thickness = 2, color_noise_dist = 1):
    # needs to be color
    x, y, r = pupil
    r *= radius_ratio
    h, w, colors = frame.shape
    y_mask, x_mask = np.ogrid[-x:w-x, -y:h-y]
    mask_small = x_mask*x_mask + y_mask*y_mask <= (r+mask_thickness)**2 
    mask_big = x_mask*x_mask + y_mask*y_mask >= r**2
    mask = reduce(np.logical_and, [mask_small, mask_big])
    # average_colors = [np.median(frame[mask][i]) for i in range(colors)]

    color_hists = []

    for i in range(colors):
        colored_frame = frame[:, :, i]
        average = np.mean(colored_frame[mask])
        colored_frame = cv2.equalizeHist(colored_frame)
        ret, colored_frame = cv2.threshold(colored_frame, average - color_noise_dist, 255, cv2.THRESH_BINARY)
        ret, colored_frame = cv2.threshold(colored_frame, average + color_noise_dist, 255, cv2.THRESH_BINARY)
        color_hists.append(colored_frame)
        colored_frame[mask] = 255
        # print(colored_frame.shape)
        cv2.imshow("this_color_hist", colored_frame)
        # print(colored_frame)
        cv2.waitKey(20)
        time.sleep(1)
    color_hist = np.max(color_hists, 0)

    # cv2.imshow("color_hist", color_hist)
    # cv2.waitKey(20)
    # time.sleep(10)

    x, y, r = get_circle_from_hist(color_hist)
    # print("iris found at %d, %d with radius %r" % (x, y, r))

    return (x, y, r)

def get_pupil_from_eye(frame):
    pupil_frame = cv2.equalizeHist(frame)
    ret, pupil_frame = cv2.threshold(pupil_frame, 15, 255, cv2.THRESH_BINARY)

    x, y, r = get_circle_from_hist(pupil_frame)
    # print("pupil found at %d, %d with radius %r" % (x, y, r))

    return (x, y, r)

def get_circle_from_hist(hist):
    window_close = np.ones((5,5),np.uint8)
    window_open = np.ones((2,2),np.uint8)
    window_erode = np.ones((2,2),np.uint8)
    hist = cv2.morphologyEx(hist, cv2.MORPH_CLOSE, window_close)
    hist = cv2.morphologyEx(hist, cv2.MORPH_ERODE, window_erode)
    hist = cv2.morphologyEx(hist, cv2.MORPH_OPEN, window_open)
    # cv2.imshow('pupil_frame', pupil_frame)
    threshold = cv2.inRange(hist,250,255)
    _, contours, hierarchy = cv2.findContours(threshold,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    # cx, cy = int(center['m10']/center['m00']), int(center['m01']/center['m00'])
    # print(contours)
    circle = get_circle_from_contours(contours)

    return circle

def get_circle_from_contours(contours, circles_ratio = 2):
    areas = [cv2.contourArea(contour) for contour in contours]
    info_of_enclosing_circles = [(x, y, radius, np.pi*radius**2) for (x, y), radius in map(lambda x: cv2.minEnclosingCircle(x), contours)]
    close_to_circles = [circle_info for i, circle_info in enumerate(info_of_enclosing_circles) if areas[i] > 0 and circle_info[3]/areas[i] < circles_ratio]
    sorted_circles = sorted(close_to_circles, key = lambda x: x[3])
    if len(sorted_circles) > 1:
        pupil = map(int, sorted_circles[-2][:3])
    else:
        pupil = map(int, sorted_circles[0][:3])

    return pupil

def find_breaking_point(stamps, rates, threshold = 350):
    # see between which peaks in flashlight brightness correspond
    # to rates not bing sustained at 0
    count = len(rates)
    f = lambda x: (rates[x] - rates[x+2]) / 2
    deriv = list(map(f, range(count - 2)))
    # print(deriv)
    breakpoints = (np.array(stamps)[:, 0] * FPS).astype(np.int)
    slow_slope = np.abs(deriv) < threshold
    # print(slow_slope)
    # breakpoint_bounds = np.array(list(zip(range(len(breakpoints) - 2), range(1, count - 1))))
    breakpoint_bounds = np.array([[breakpoints[i], breakpoints[i+1]] for i in range(len(breakpoints) - 1)])
    # print(breakpoint_bounds)
    stab = np.array([slow_slope[x:y].any() for x, y in breakpoint_bounds])
    # print(stab)
    spot = np.where(stab == False)[0][0]

    print(spot)

    return spot

if __name__ == "__main__":
    print(sys.argv)
    video_location = sys.argv[1]
    if len(sys.argv) > 2:
        save_location = sys.argv[2]
        stamps = ast.literal_eval(sys.argv[3])
    ratios, rates = get_dilation_rates_from_video(video = video_location, save_to = save_location, stamps = stamps)
    # print(ratios)
    # print(rates)
    # print(stamps)
    # find_breaking_point(stamps, rates)
