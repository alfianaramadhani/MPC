#referensi kode AutonomousCarV0.6.py, MPC2, dan spatial_models3
#kode untuk imageprocessing dari folder mpc/imageprocess3.py

import cv2
import numpy as np
import os
from scipy import optimize
from matplotlib import pyplot as plt, cm, colors

import math
from Model4 import BicycleModel
from Kontroller4 import MPC
from scipy import sparse #membuat matriks untuk untuk perhitungan MPC

import csv
import time
from datetime import datetime

#LIBRARY KOMUNIKASI
import smbus2
from pylibftdi import Device
import threading

##KODE KONTROL UNTUK MOTOR DAN STEERING##
##########  MULAI  ######################
# Constants
I2C_BUS = 1
DEVICE_ADDRESS = 0x32
DEVICE_REGISTER = 0x00

# Setup I2C bus
bus = smbus2.SMBus(I2C_BUS)

CENTER_STEERING_ANGLE = 21  # Sesuaikan dengan nilai yang diinginkan
# Global variables and lock
global stop_flag, start_flag, center_steering
stop_flag = False
start_flag = False
center_steering = False
kec = 0
kec_measured = 0
angle = CENTER_STEERING_ANGLE
steering_angle_deg = 0
steering_angle_rad = 0
frame = None
frame_lock = threading.Lock()

class CustomDevice(Device):
    def __init__(self):
        super().__init__()
        self.baudrate = 19200
        
def set_buffer(data_type, speed, steering, brake, p, i, d):
    buffer = bytearray(13)
    if data_type == "start":
        buffer[0] = 0x01
        for i in range(1, 12):
            buffer[i] = 0x00
        buffer[12] = 0x01
    elif data_type == "stop":
        buffer[0] = 0x01
        for i in range(1, 13):
            buffer[i] = 0x00
    elif data_type == "set":
        data_values = [speed, steering, brake, p, i, d]
        for index, value in enumerate(data_values):
            integer_part = int(abs(value))
            fractional_part = int((abs(value) * 100) % 100)
            buffer[2 * index + 1] = min(integer_part, 255)
            buffer[2 * index + 2] = min(fractional_part, 99)

        buffer[0] = 0x02
        buffer[12] = 0x00

    return buffer
    
def send_data(speed, pwm, command):
    with CustomDevice() as dev:
        speed = int(speed) & 0xFF
        pwm = int(pwm) & 0xFF
        command_high = (command >> 8) & 0xFF
        command_low = command & 0xFF

        data_to_send = bytes([speed, pwm, command_high, command_low])
        dev.write(data_to_send)

def write_i2c_data(data):
    try:
        msg = smbus2.i2c_msg.write(DEVICE_ADDRESS, data)
        bus.i2c_rdwr(msg)
        read = smbus2.i2c_msg.read(DEVICE_ADDRESS, 1)
        bus.i2c_rdwr(read)
        response = list(read)[0]
        return response == 49
    except Exception as e:
        print(f"I2C Error: {e}")
        return False
        
def data_sending_thread():
    while True: 
        send_data(kec, 255, 0xABCD)

def stop_system():
    global stop_flag
    stop_flag = True
    print("Stopping system...")
    set_data = set_buffer("set", 0, 0, 0, 0, 0, 0)
    write_i2c_data(set_data)
    stop_data = set_buffer("stop", 0, 0, 0, 0, 0, 0)
    send_data(0, 255, 0xABCD)
    time.sleep(1)
    write_i2c_data(stop_data)
    
def center_steering_system():
    global center_steering 
    center_steering = True
    print("Centering steering...")
    set_data = set_buffer("set", 0, CENTER_STEERING_ANGLE, 0, 0, 0, 0)
    write_i2c_data(set_data)

def read_serial_data(): #membaca data dari arduino
    global kec_measured
    try:
        with CustomDevice() as dev:
            dev.ftdi_fn.ftdi_set_line_property(8, 1, 0)  # 8 bits, 1 stop bit, no parity
            dev.ftdi_fn.ftdi_set_baudrate(19200)
            
            buffer = b''
            while True:
                data = dev.read(1)
                if not data:
                    time.sleep(0.01)
                    continue
                
                if data == b'\n':  # End of line
                    try:
                        decoded = buffer.decode('ascii')
                        values = decoded.strip().split(',')
                        if len(values) >= 3:
                            kec_measured = float(values[0])  # speed in km/h
                            setpoint = float(values[1])     # setpoint
                            pid_output = float(values[2])   # PID output
                            print(f"Measured Speed: {kec_measured:.2f} km/h, Setpoint: {setpoint:.2f}, PID Output: {pid_output:.2f}")
                    except Exception as e:
                        print(f"Error parsing serial data: {e}")
                    
                    buffer = b''
                else:
                    buffer += data
    except Exception as e:
        print(f"Serial reading error: {e}")
        
##KODE KONTROL UNTUK MOTOR DAN STEERING##
##########  SELESAI  ####################

##KODE UNTUK IMAGE PROCESSING##
######## MULAI ################
#skala meter per pixel
ym_per_pix = 1.51 / 480 #mengasumsikan 1,51 meter mewakili 480 pixel
xm_per_pix = 1.8 / 640 #mengasumsikan 1.8 meter mewakili 640 pixel

class ImprovedLaneDetection:
    def __init__(self):
        self.tl, self.tr, self.br, self.bl =  [140, 132], [490, 143], [622, 268], [7, 249]
        self.src_points = np.float32([self.tl, self.tr, self.br, self.bl])

        #self.src_points = np.float32([
            #[119, 237], #top left
            #[521, 237], #top right
            #[631, 445], #bottom right
            #[6, 435] #bottom left
        #    tl,tr,br,bl
        #])
        
        self.dst_points = np.float32([
            [0, 0], [640, 0], [640, 480], [0, 480]
        ])
        self.nwindows = 9
        #self.margin = 100
        #self.minpix = 50
        
        self.margin = 65
        self.minpix = 50
        
        # Create output directories if they don't exist
        if not os.path.exists("output_frames"):
            os.makedirs("output_frames")
        if not os.path.exists("output_data"):
            os.makedirs("output_data")
            
        # Initialize CSV file with headers
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_filename = f"output_data/lane_data_{timestamp}.csv"
        with open(self.csv_filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Frame', 'Kecepatan', 'Y0X_Pixel', 'Y239X_Pixel', 'Y479X_Pixel', 'Y0X_Meter', 'Y239X_Meter', 'Y479X_Meter', 'errorY_0_Xmeter', 'errorY_239_Xmeter', 'errorY_479_Xmeter','Radius', 'Curvature', 'Y0(meter)', 'Y239(meter)', 'Y479(meter)', 'Angle_MPC(rad)','Angle_MPC(deg)','Konversi_MPC2Angle', 'Angle','FPS', 'km/h'])
            
    def preprocess_image(self, img):
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        l_channel = hls[:, :, 1]
        l_binary = np.zeros_like(l_channel) #membuat array kosong deukuran l_channel, tetapi semua nilainya 0 (hitam)
        l_binary[(l_channel >= 180) & (l_channel <= 255)] = 255 #memilih pixel dengan nilai lightness antara 180 dan 255, kemudian piksel yang memenuhi syarat diubah menjadi 255 (putih)
        return l_binary

    def bird_eye_transform(self, img):
        img_size = (img.shape[1], img.shape[0])
        M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        return cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    def compute_histogram(self, binary_warped, frame_count):
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
        plt.figure()
        plt.plot(histogram)
        plt.title("Lane Histogram")
        plt.xlabel("Pixel Position")
        plt.ylabel("Intensity Sum")
        plt.savefig(f"output_frames/frame_{frame_count:04d}_histogram.jpg")
        plt.close()
        return histogram

    def sliding_window(self, binary_warped):
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
        midpoint = np.int32(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        window_height = np.int32(binary_warped.shape[0] / self.nwindows)
        nonzero = binary_warped.nonzero()
        nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])
        leftx_current, rightx_current = leftx_base, rightx_base
        left_lane_inds, right_lane_inds = [], []
        
        for window in range(self.nwindows):
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low, win_xleft_high = leftx_current - self.margin, leftx_current + self.margin
            win_xright_low, win_xright_high = rightx_current - self.margin, rightx_current + self.margin
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            if len(good_left_inds) > self.minpix:
                leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:
                rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))
        
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return out_img, leftx, lefty, rightx, righty
        
    def calculate_curvature(self, binary_warped, leftx, lefty, rightx, righty):  
        # Cek jika array kosong atau tidak sama panjang          
        #if (len(leftx) == 0 or len(lefty) == 0 or len(rightx) == 0 or len(righty) == 0 or
        #    len(leftx) != len(lefty) or len(rightx) != len(righty)):
        #    return 0.0, "Tidak Terdeteksi"

        # Menggunakan nilai konversi piksel ke meter
        ym_per_pix = 1.51 / 480 #mengasumsikan 1,51 meter mewakili 480 pixel
        xm_per_pix = 1.8 / 640 #mengasumsikan 1.8 meter mewakili 640 pixel
        
     
        
        # Membalik array untuk mencocokkan orientasi dari atas ke bawah dalam y
        # seperti pada kode 2 (measure_lane_curvature)
        leftx = leftx[::-1]
        lefty = lefty[::-1]
        rightx = rightx[::-1]
        righty = righty[::-1]
        
        # Buat ploty terpisah untuk left dan right lane
        left_ploty = np.linspace(0, binary_warped.shape[0]-1, len(leftx))
        right_ploty = np.linspace(0, binary_warped.shape[0]-1, len(rightx))
        
        # Fit polinomial baru ke x, y dalam space dunia nyata
        left_fit_cr = np.polyfit(left_ploty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(right_ploty*ym_per_pix, rightx*xm_per_pix, 2)
        
        # Gunakan nilai y maksimum, sesuai dengan bagian bawah gambar
        left_y_eval = np.max(left_ploty)
        right_y_eval = np.max(right_ploty)
        
        # Hitung jari-jari kurvatur baru
        left_curverad = ((1 + (2 * left_fit_cr[0] * left_y_eval * ym_per_pix + left_fit_cr[1]) ** 2) * 1.5) / np.absolute(2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * right_y_eval * ym_per_pix + right_fit_cr[1]) ** 2) * 1.5) / np.absolute(2 * right_fit_cr[0])
        
        # Tentukan apakah itu kurva kiri atau kanan
        if len(leftx) > 1 and (leftx[0] - leftx[-1] > 60):
            curve_direction = 'Left Curve'
        elif len(leftx) > 1 and (leftx[-1] - leftx[0] > 60):
            curve_direction = 'Right Curve'
        else:
            curve_direction = 'Straight'
        
        # Debug output
        #print(f"DEBUG: left_fit_cr = {left_fit_cr}, right_fit_cr = {right_fit_cr}")
        #print(f"DEBUG: left_curverad = {left_curverad}, right_curverad = {right_curverad}")
        
        # Mengembalikan rata-rata radius dan arah kurva
        return (left_curverad + right_curverad) / 2.0, curve_direction
        
    def visualize_curvature(self, binary_warped, leftx, lefty, rightx, righty):
        #if (len(leftx) == 0 or len(lefty) == 0 or len(rightx) == 0 or len(righty) == 0 or
        #    len(leftx) != len(lefty) or len(rightx) != len(righty)):
        #    return np.zeros((binary_warped.shape[0], binary_warped.shape[1], 3), dtype=np.uint8), {}

        # Fit polynomial to left and right lane lines
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        
        # Generate points to plot
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        # Create color visualization image
        curve_img = np.zeros((binary_warped.shape[0], binary_warped.shape[1], 3), dtype=np.uint8)
        
        # Draw left lane (green)
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        cv2.polylines(curve_img, np.int_([pts_left]), isClosed=False, color=(0,255,0), thickness=10)
        
        # Draw right lane (red)
        pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty]))])
        cv2.polylines(curve_img, np.int_([pts_right]), isClosed=False, color=(0,0,255), thickness=10)
        
        # Calculate and draw center line
        center_fitx = (left_fitx + right_fitx) / 2
        
        # Return the center points as a dictionary for easier access
        center_points = {int(y): int(x) for y, x in zip(ploty, center_fitx)}
        
        # Draw center line (white)
        pts_center = np.array([np.transpose(np.vstack([center_fitx, ploty]))])
        cv2.polylines(curve_img, np.int_([pts_center]), isClosed=False, color=(255,255,255), thickness=5)
        
        return curve_img, center_points
        
        
##KODE UNTUK IMAGE PROCESSING##
######## SELESAI ##############

def ModelPredictiveControl ():
    ##KODE UNTUK SETTING MPC##
    ####### MULAI ############
    model = BicycleModel(length=0.7, width=0.75) #dalam meter

    # Pembobotan objective function 
    N = 3                               # Panjang horizon prediksi 
    Q = sparse.diags([1.0, 0.0, 0.0])   # Bobot kesalahan lateral prediksi 
    R = sparse.diags([0.5, 0.0])        # Bobot penalti input 
    QN = sparse.diags([1.2, 0.0, 0.0])  # Bobot kesalahan lateral titik ujung horizon 

    #Kendala
    #delta_max = 20.95  # batas sudut kemudi, derajat.
    delta_max = 0.366  # batas sudut kemudi, radian.
    v_max = 1 #batas kecepatan (m/s)--> kalau dalam kode raspi itu pake rentang kec = 7 dikonversi menjadi 3.61 km/h kemudian menjadi 1 m/s
    ay_max = 0.5 #batas akselerasi, m/s^2
    InputConstraints = {'umin': np.array([0.0, -np.tan(delta_max)/model.length]),
                        'umax': np.array([v_max, np.tan(delta_max)/model.length])}
    StateConstraints = {'xmin': np.array([-np.inf, -np.inf, -np.inf]),
                        'xmax': np.array([np.inf, np.inf, np.inf])}
    mpc = MPC(model, N, Q, R, QN, StateConstraints, InputConstraints, ay_max)
    
    v_ref = 0.875  # kecepatan referensi(m/s)--> kalau dalam raspi itu pake rentang kec = 5 --> 3.15 km/jam --> 0.875 m/s

    # Definisi koordinat y lintasan, dan delta_s 
    #yy = [0*ym_per_pix, 239*ym_per_pix, 479*ym_per_pix] #diambil di titik y ke-0, 239, dan 479
    yy = [0*ym_per_pix, 236*ym_per_pix, 477*ym_per_pix] #diambil di titik y ke-0, 236, dan 477
    delta_s = yy[1]-yy[0] 
    
    return mpc, delta_s, yy, v_ref
    ##KODE UNTUK SETTING MPC##
    ####### SELESAI ############
    

def main():
    global steering_angle_deg, steering_angle_rad, angle, kec, kec_measured, mpc_active, start_flag #purepersuit_active
    cap = cv2.VideoCapture(0)  # Menggunakan kamera langsung
    #cap = cv2.VideoCapture("video.mp4") #video
    if not cap.isOpened():
        print("Gagal membuka kamera.")
        exit()
    detector = ImprovedLaneDetection()
    frame_count = 0
    mpc, delta_s, yy, v_ref = ModelPredictiveControl()
    
    try:
        start_data = set_buffer("start", 0, 0, 0, 0, 0, 0)
        write_i2c_data(start_data)
        set_data = set_buffer("set", 0, 20, 0, 0, 0, 0)
        write_i2c_data(set_data)
    except Exception as e:
        print(f"Failed to open I2C port: {e}")
        return

    threading.Thread(target=data_sending_thread, daemon=True).start()
    threading.Thread(target=read_serial_data, daemon=True).start()  # Start data reading thread
    kec = 0
    kec_measured = 0
    mpc_active = False  # Pastikan MPC tidak berjalan sampai tombol ditekan
    #purepersuite_active = False # Pastikan Pure Persuit tidak berjalan sampai tombol ditekan
    
    prev_time = time.time()  # <-- Inisialisasi prev_time di sini (ini untuk fps)
    
    while True:
        try :
            #Untuk Menghitung FPS :
            current_time = time.time()
            if current_time - prev_time > 0:
                fps = 1.0 / (current_time - prev_time)
            prev_time = current_time  # <-- Perbarui prev_time setiap iterasi
            
            ret, frame = cap.read()
            if not ret or stop_flag:
                break
            
            frame = cv2.resize(frame, (640, 480))
            binary = detector.preprocess_image(frame)
            warped = detector.bird_eye_transform(binary)
            window_img, leftx, lefty, rightx, righty = detector.sliding_window(warped)
            curve_img, center_points = detector.visualize_curvature(warped, leftx, lefty, rightx, righty)
            curvature = detector.calculate_curvature(warped, leftx, lefty, rightx, righty) #calculate curvature
            #histogram = detector.compute_histogram(warped, frame_count) #ini yang bikin fpsnya jeblok
            
            #x_values = {0: None, 239: None, 479: None} # Initialize x_values with default None values
            #xx = [(center_points[y] - 320) * xm_per_pix for y in [0, 239, 479] if y in center_points] # Ambil titik-titik tengah jalur pada Y = 0, 239, 479
            x_values = {0: None, 236: None, 477: None} # Initialize x_values with default None values
            xx = [(center_points[y] - 320) * xm_per_pix for y in [0, 236, 477] if y in center_points] # Ambil titik-titik tengah jalur pada Y = 0, 239, 479

            # Jika jumlah titik tidak cukup, isi dengan nilai default (0.0)
            while len(xx) < 3:
                xx.append(0.0)
            
            #Kontrol input dari keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('j'):  # Mulai mobil
                start_flag = True
                kec = 5
                print("Mobil mulai berjalan...")
            elif key == ord('c'):  # Tengahkan steering
                center_steering_system()
            elif key == ord('m'):  # Aktifkan MPC
                mpc_active = True
                print("MPC Aktif...")
            #elif key == ord('p'):  # Aktifkan Pure Persuit
            #    purepersuit_active = True
            #    print("Pure Persuit Aktif...")
            elif key == ord('q'):  # Hentikan sistem
                stop_system()
                break

            # Hanya jalankan MPC jika tombol 'm' sudah ditekan
            if mpc_active and len(xx) == 3:
                #mpc.kom(delta_s, 1 / curvature if curvature != 0 else 0, xx)
                #mpc.kom(delta_ s, 1 / curvature, xx) 
                mpc.reff(delta_s, 1 / curvature[0], v_ref, xx)
                mpc.get_control()
                
                #Konversi hasil optimasi steering di MPC ke kontrol steering mobil asli (karena MPC memberikan hasil dalam rentang -20.95 dan 20.95 derajat) tapi aslinya outputnya radian
                #sedangkan di mobil itu maksimal kiri 0 derajat, tengah sekitar 27 derajat, dan maksimal kanan 45 derajat
    
                #batas maksimal -20.95 dan 20.95 derajat
                mpc_control1 = np.clip(mpc.current_control[1], -0.366, 0.366) #dalam radian
                
                ## Ubah ke rentang 0° - 40° (dalam bentuk radian)
                steering_angle_rad = ((mpc_control1 + 0.366) / 0.732) * np.radians(19) #karena angle aku setting di 21 derajat, maka dikalikan 19 (40-21 = 19)
                
            
            #if purepersuit_active and len(xx) == 3:
                ##Pure Pursuit - Start ##
            #    alpha = math.atan2(xx[1]-xx[0], yy[1])# - 1.5708
            #    delta = math.atan2(2.0 * 0.23 * math.cos(math.radians(90)- alpha) / yy[3], 1.0)
                ##Pure Pursuit - End  ##
            
            steering_angle_deg = np.degrees(steering_angle_rad)
            angle = 21 + steering_angle_deg 

            
            set_data = set_buffer("set", 3.4, angle, 100, 0, 0, 0)
            if not write_i2c_data(set_data):
                write_i2c_data(set_data)
            
            print("angle:", angle)
            #print("FPS:", round(fps, 2))  # Print FPS
            
            # Update x_values with actual values if they exist in center_points
            for y in x_values.keys():
                if y in center_points:
                    x_values[y] = center_points[y]
                    # Draw on the image if we have a value
                    if x_values[y] is not None:
                        cv2.putText(curve_img, f"x: {x_values[y]}", 
                                   (x_values[y], y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.5, (255, 255, 255), 2)
                        
            mpcsteeringangledeg = np.degrees(mpc.current_control[1])
            
            # Save data to CSV
            with open(detector.csv_filename, 'a', newline='') as file:
                writer = csv.writer(file)
                
                curvature_value = curvature if isinstance(curvature, (int, float)) else 0.0  # Pastikan curvature berupa angka

                writer.writerow([
                    frame_count, kec, 
                    #x_values[0], x_values[239], x_values[479], 
                    #x_values[0]*xm_per_pix , x_values[239]*xm_per_pix , x_values[479]*xm_per_pix, 
                    x_values[0], x_values[236], x_values[477], 
                    x_values[0]*xm_per_pix , x_values[236]*xm_per_pix , x_values[477]*xm_per_pix,
                    xx[0], xx[1], xx[2], 
                    curvature, #f"{(1/curvature_value):.2f}" if curvature_value != 0 else "0.00",
                    1/curvature[0],
                    yy[0],yy[1],yy[2], 
                    mpc.current_control[1], mpcsteeringangledeg,
                    steering_angle_deg, angle,f"{fps:.1f}",
                    kec_measured
                ])
                
            # Menampilkan Di Layar
            cv2.putText(curve_img, f"Sudut: {angle}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(curve_img, f"MPC: {steering_angle_deg}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            #cv2.putText(curve_img, f"KecMPC: {mpc.current_control[0]}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            #cv2.putText(curve_img, f"km/j: {kec_measured}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            #Memunculkan Layar
            cv2.circle(frame, tuple(detector.tl), 5, (0,0,255), -1)
            cv2.circle(frame, tuple(detector.bl), 5, (0,0,255), -1)
            cv2.circle(frame, tuple(detector.tr), 5, (0,0,255), -1)
            cv2.circle(frame, tuple(detector.br), 5, (0,0,255), -1)
            
            cv2.imshow ("original", frame)
            cv2.imshow ("sliding_window", window_img)
            cv2.imshow ("curvature", curve_img)
            
            cv2.imwrite(f"output_frames/frame_{frame_count:04d}_original.jpg", frame)
            #cv2.imwrite(f"output_frames/frame_{frame_count:04d}_hls.jpg", binary)
            cv2.imwrite(f"output_frames/frame_{frame_count:04d}_bird_eye.jpg", warped)
            cv2.imwrite(f"output_frames/frame_{frame_count:04d}_sliding_window.jpg", window_img)
            cv2.imwrite(f"output_frames/frame_{frame_count:04d}_curvature.jpg", curve_img)
            
            frame_count += 1
            
        except Exception as e:
            print(f"Error terjadi: {e}")
            stop_system()
            break

    cap.release()
    cv2.destroyAllWindows()
    stop_system()

if __name__ == "__main__":
    main()
    # berhentikan seluruh aktuator ketika selesai
    stop_data = set_buffer("stop", 0, 0, 0, 0, 0, 0)
    write_i2c_data(stop_data)
    send_data(0, 255, 0xABCD)
