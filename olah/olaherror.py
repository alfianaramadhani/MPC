import cv2
import numpy as np
import math
import time
import csv

ym_per_pix = 1.51 / 480
xm_per_pix = 1.8 / 640

class CoordinateStore:
    def __init__(self):
        self.points = []

    def select_point(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(img, (x, y), 3, (255, 0, 0), -1)
            self.points.append((x, y))

def angle(P2, P1):
    rad = math.atan2((P2[0] - P1[0]) * xm_per_pix, (P2[1] - P1[1]) * ym_per_pix)
    deg = math.degrees(rad)
    return rad, deg

# instantiate class
coordinateStore1 = CoordinateStore()

# Prepare image and window
timestr = time.strftime("%Y%m%d-%H%M%S")
img = cv2.imread('frame_0_bird_eye.jpg')
cv2.namedWindow('image')
cv2.setMouseCallback('image', coordinateStore1.select_point)

# Open a single CSV file for writing
csvfile = open("output_data_" + timestr + ".csv", "w", newline='')

# Create a CSV writer
csv_writer = csv.writer(csvfile)

# Write header for the CSV file
csv_writer.writerow(['Angle (deg)', 'Lateral Error', 'Image Index'])

#i = 300 #ini karena Pak Azis diloncatin setiap 300 frame
i = 0

while True:
    cv2.imshow('image', img)
    k = cv2.waitKey(20) & 0xFF
    if k == ord('z'):
        img = cv2.imread('frame_' + str(i) + '_bird_eye.jpg')
    if np.any(img) == None:
        a = 0
        while np.any(img) == None:
            a += 1
            if a == 1000:
                print('stop skipping')
                csvfile.close()
                break
            i += 1
            img = cv2.imread('outputbirdview-pic-' + str(i) + '.jpg')
            print('skipping order', i - 1)
        i -= 1
    if k == ord('a'):
        alpharad, alphadeg = angle(coordinateStore1.points[-2], coordinateStore1.points[-1])
        print(alphadeg)
        continue
    if k == ord('s'): #jika kedua garis diketahui
        xtengah = ((coordinateStore1.points[-2][0]+coordinateStore1.points[-1][0])/2)
        erlat = (xtengah-320)*xm_per_pix
        print(erlat)
    if k == ord('d'): #jika hanya garis kanan yang diketahui
        erlat = (coordinateStore1.points[-2][0]-640)*xm_per_pix   
        print(erlat)
        continue
    if k == ord('f'): #jika hanya garis kiri yang diketahui
        erlat = (coordinateStore1.points[-2][0]-0)*xm_per_pix
        print(erlat)
    if k == ord('g'):
        # Write data to CSV in one row
        csv_writer.writerow([alphadeg, erlat, i])  # Write angle, lateral error, and image index
        print(alphadeg, erlat, i)
        i += 1
    if k == ord('x'):
        print(coordinateStore1.points[-2][0], coordinateStore1.points[-1][0])
        print(coordinateStore1.points[-2][0] - coordinateStore1.points[-1][0])
        print((coordinateStore1.points[-2][0] - coordinateStore1.points[-1][0]) * xm_per_pix)
    if k == 27:
        break

cv2.destroyAllWindows()
# Close CSV file
csvfile.close()
