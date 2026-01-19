import cv2
import mediapipe as mp
import math
import os
import numpy as np
import sys

# --- 1. ตั้งค่า MediaPipe ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# --- ฟังก์ชันคำนวณมุม ---
def calculate_angle(a, b, c):
    a = np.array(a) 
    b = np.array(b) 
    c = np.array(c) 
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# --- ฟังก์ชันวาดข้อความพร้อมขอบ ---
def draw_text_with_outline(img, text, pos, font, scale, text_color, outline_color, thickness, outline_thickness):
    cv2.putText(img, text, pos, font, scale, outline_color, outline_thickness)
    cv2.putText(img, text, pos, font, scale, text_color, thickness)

# --- ตัวแปรเก็บค่า ---

# ==========================================
# 📏 CALIBRATION SETTING
# ==========================================
pixels_per_cm = 5.0  # calibration pixel สำหรับใส่ค่ากรณีคำนวณให้ออกมาเป็น cm

# 1. Shoulder
peaks_shoulder = []         
temp_max_shoulder = 0       
avg_max_shoulder = 0        
prev_shoulder = 0 

# 2. Feet Distance (Foot Length)
peaks_foot = []      
temp_max_foot = 0    
avg_max_foot = 0     
prev_foot = 0 

# 3. Knee Angle (MAX)
peaks_knee = []          
temp_max_knee = 0        
avg_max_knee = 0
prev_knee = 0

# 4. Feet Height
peaks_fh = []        
temp_max_fh_L = -100 
temp_max_fh_R = -100 
avg_max_fh = 0       
prev_fh_L = 0
prev_fh_R = 0

# --- 2. ระบุไฟล์ VDO ---
video_path = "D:\Desktop\Program\Gait/walk_vdo.mp4" 

if not os.path.exists(video_path):
    print(f"❌ Error: หาไฟล์ไม่เจอที่ -> {video_path}")
    sys.exit()

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Error: เปิดไฟล์วิดีโอไม่ได้")
    sys.exit()

print(f"🎥 กำลังคำนวณ... (สูตร Total Risk ใหม่)")
print("กด 'q' เพื่อปิดโปรแกรม")

last_final_image = None 

while cap.isOpened():
    success, image = cap.read()
    
    if not success:
        if last_final_image is not None:
            # ข้อความจบวิดีโอ (Calibration Info)
            info_text = f"Calibration: {pixels_per_cm} Pixel/CM"
            draw_text_with_outline(last_final_image, info_text, 
                                   (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), (255, 255, 255), 2, 5)
            
            cv2.imshow('Side-by-Side Analysis', last_final_image)
            while True:
                key_end = cv2.waitKey(100) 
                if key_end == ord('q') or key_end == ord('Q') or key_end == 27: 
                    cap.release()
                    cv2.destroyAllWindows()
                    sys.exit()
                if cv2.getWindowProperty('Side-by-Side Analysis', cv2.WND_PROP_VISIBLE) < 1:
                    cap.release()
                    sys.exit()
        break 

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    h, w, c = image.shape

    # ---------------------------------------------------------
    # 🎨 ส่วนคำนวณ (Calculation)
    # ---------------------------------------------------------
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # 1. Shoulder (Max)
        x1, y1 = int(landmarks[11].x * w), int(landmarks[11].y * h)
        x2, y2 = int(landmarks[23].x * w), int(landmarks[23].y * h)
        distance_shoulder_px = abs(x2 - x1)

        if distance_shoulder_px < prev_shoulder - 2: 
            if temp_max_shoulder > 5:
                val_cm = temp_max_shoulder / pixels_per_cm
                peaks_shoulder.append(val_cm)
                peaks_shoulder.sort(reverse=True)
                peaks_shoulder = peaks_shoulder[:5] 
                if len(peaks_shoulder) > 0:
                    avg_max_shoulder = sum(peaks_shoulder) / len(peaks_shoulder)
                temp_max_shoulder = 0 
        if distance_shoulder_px > temp_max_shoulder:
            temp_max_shoulder = distance_shoulder_px
        prev_shoulder = distance_shoulder_px

        cv2.circle(image, (x1, y1), 6, (0, 0, 255), cv2.FILLED) 
        cv2.circle(image, (x2, y2), 6, (0, 0, 255), cv2.FILLED)
        cv2.line(image, (x1, y1), (x2, y1), (0, 255, 255), 2)
        cv2.line(image, (x2, y1), (x2, y2), (255, 0, 0), 1)

        # 2. Feet Distance (Foot Length: Heel to Toe)
        x3, y3 = int(landmarks[29].x * w), int(landmarks[29].y * h) # ส้นเท้าซ้าย
        x4, y4 = int(landmarks[31].x * w), int(landmarks[31].y * h) # ปลายเท้าซ้าย
        
        dist_foot_px = math.sqrt((x4 - x3)**2 + (y4 - y3)**2)

        if dist_foot_px < prev_foot - 2: 
            if temp_max_foot > 5: 
                val_cm = temp_max_foot / pixels_per_cm
                peaks_foot.append(val_cm)
                peaks_foot.sort(reverse=True)
                peaks_foot = peaks_foot[:5] 
                if len(peaks_foot) > 0:
                    avg_max_foot = sum(peaks_foot) / len(peaks_foot)
                temp_max_foot = 0 
        if dist_foot_px > temp_max_foot:
            temp_max_foot = dist_foot_px
        prev_foot = dist_foot_px

        cv2.circle(image, (x3, y3), 6, (0, 255, 0), cv2.FILLED) 
        cv2.circle(image, (x4, y4), 6, (0, 255, 0), cv2.FILLED) 
        cv2.line(image, (x3, y3), (x4, y4), (255, 0, 255), 2)   

        # 3. Knee Angle (Max)
        k1 = [landmarks[23].x * w, landmarks[23].y * h] 
        k2 = [landmarks[25].x * w, landmarks[25].y * h] 
        k3 = [landmarks[27].x * w, landmarks[27].y * h] 
        angle_knee = calculate_angle(k1, k2, k3)

        if angle_knee < prev_knee - 2: 
            if temp_max_knee > 50: 
                peaks_knee.append(temp_max_knee)
                peaks_knee.sort(reverse=True) 
                peaks_knee = peaks_knee[:5]   
                if len(peaks_knee) > 0:
                    avg_max_knee = sum(peaks_knee) / len(peaks_knee)
                temp_max_knee = 0 
        if angle_knee > temp_max_knee:
            temp_max_knee = angle_knee
        prev_knee = angle_knee

        cv2.circle(image, (int(k2[0]), int(k2[1])), 8, (0, 255, 0), cv2.FILLED) 
        draw_text_with_outline(image, f'{int(angle_knee)}', (int(k2[0])+10, int(k2[1])), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), (0,0,0), 2, 4)
        
        # 4. Feet Height (Max)
        heel_L_y = landmarks[29].y * h
        toe_L_y = landmarks[31].y * h
        height_L_px = heel_L_y - toe_L_y

        heel_R_y = landmarks[30].y * h
        toe_R_y = landmarks[32].y * h
        height_R_px = heel_R_y - toe_R_y

        if height_L_px < prev_fh_L - 0.5: 
            if temp_max_fh_L > -20: 
                val_cm = temp_max_fh_L / pixels_per_cm
                peaks_fh.append(val_cm)
                peaks_fh.sort(reverse=True)
                peaks_fh = peaks_fh[:5] 
                if len(peaks_fh) > 0:
                    avg_max_fh = sum(peaks_fh) / len(peaks_fh)
                temp_max_fh_L = -100 
        if height_L_px > temp_max_fh_L:
            temp_max_fh_L = height_L_px
        prev_fh_L = height_L_px

        if height_R_px < prev_fh_R - 0.5:
            if temp_max_fh_R > -20:
                val_cm = temp_max_fh_R / pixels_per_cm
                peaks_fh.append(val_cm)
                peaks_fh.sort(reverse=True)
                peaks_fh = peaks_fh[:5]
                if len(peaks_fh) > 0:
                    avg_max_fh = sum(peaks_fh) / len(peaks_fh)
                temp_max_fh_R = -100
        if height_R_px > temp_max_fh_R:
            temp_max_fh_R = height_R_px
        prev_fh_R = height_R_px

    # ---------------------------------------------------------
    # 🖼️ สร้างหน้าต่าง และ ตกแต่ง (Background สีเทา)
    # ---------------------------------------------------------
    panel_width = 450
    min_height = 800 
    final_h = max(h, min_height) 
    
    # สร้างภาพพื้นหลัง (สีเทาอ่อน BGR: 240, 240, 240)
    final_image = np.ones((final_h, w + panel_width, 3), dtype=np.uint8) * 240 
    final_image[0:h, 0:w] = image 

    # --- ตำแหน่งและสี ---
    start_x = w + 20      
    col2_x = start_x + 170
    row1_y = 60           
    row2_y = 270 

    color_head = (0, 100, 0)      
    color_data = (128, 0, 128)    
    color_risk = (0, 0, 255)      
    color_ok = (0, 128, 0)        
    
    # --- Header ---
    cv2.putText(final_image, "GAIT ANALYSIS", (start_x, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Row 1
    cv2.putText(final_image, "AVG:Shoulder (cm)", (start_x, row1_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_head, 1)
    cv2.putText(final_image, f"{avg_max_shoulder:.1f}", (start_x, row1_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color_data, 2)
    for i, p in enumerate(peaks_shoulder):
        cv2.putText(final_image, f"#{i+1}: {p:.1f}", (start_x, row1_y + 70 + (i*25)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_data, 1)

    cv2.putText(final_image, "AVG:Feet Dist (cm)", (col2_x, row1_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_head, 1)
    cv2.putText(final_image, f"{avg_max_foot:.1f}", (col2_x, row1_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color_data, 2)
    for i, p in enumerate(peaks_foot):
        cv2.putText(final_image, f"#{i+1}: {p:.1f}", (col2_x, row1_y + 70 + (i*25)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_data, 1)

    # Row 2
    cv2.putText(final_image, "AVG:Max KNEE (deg)", (start_x, row2_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_head, 1)
    cv2.putText(final_image, f"{int(avg_max_knee)}", (start_x, row2_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color_data, 2)
    for i, p in enumerate(peaks_knee):
        cv2.putText(final_image, f"#{i+1}: {int(p)}", (start_x, row2_y + 70 + (i*25)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_data, 1)

    cv2.putText(final_image, "AVG:Feet Height (cm)", (col2_x, row2_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_head, 1)
    cv2.putText(final_image, f"{avg_max_fh:.1f}", (col2_x, row2_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color_data, 2)
    for i, p in enumerate(peaks_fh):
        cv2.putText(final_image, f"#{i+1}: {p:.1f}", (col2_x, row2_y + 70 + (i*25)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_data, 1)

    # === Result Section ===
    box_y_start = row2_y + 210
    
    cv2.rectangle(final_image, (start_x, box_y_start), (w + panel_width - 10, final_h - 20), (220, 220, 220), cv2.FILLED)

    # -------------------------------------------------------------------------
    # 🔢 Calculation Logic
    # -------------------------------------------------------------------------
    sf_risk_percent = 0.0
    if len(peaks_shoulder) > 0 and avg_max_foot > 0:
        sd_shoulder = np.std(peaks_shoulder)
        denominator = avg_max_foot / 2.0
        if denominator > 0:
            if sd_shoulder > 0.5: numerator = np.median(peaks_shoulder)
            else: numerator = avg_max_shoulder
            sf_risk_percent = (numerator / denominator) * 100

    knee_risk_percent = 0.0
    if len(peaks_knee) > 0:
        sd_knee = np.std(peaks_knee)
        if sd_knee > 0.5: knee_val = np.median(peaks_knee)
        else: knee_val = np.mean(peaks_knee)
        knee_risk_percent = ((180 - knee_val) / 40.0) * 100

    fh_status_str = ""
    if len(peaks_fh) > 0:
        if min(peaks_fh) > 2.0:
            fh_status_str = "Safe"
        else:
            low_peaks = [h for h in peaks_fh if h <= 2.0]
            if len(low_peaks) > 0:
                sd_fh = np.std(low_peaks)
                if sd_fh > 0.5: fh_val = np.median(low_peaks)
                else: fh_val = np.mean(low_peaks)
                fh_risk_val_calc = ((2.0 - fh_val) / 2.0) * 100
                fh_status_str = f"{fh_risk_val_calc:.1f}%"
            else:
                fh_status_str = "Calc Err"

    # =========================================================
    # 4. Total Risk (New Formula)
    # Formula: (0.290 * Knee %) + (0.205 * S&F %)
    # =========================================================
    total_risk = (0.290 * knee_risk_percent) + (0.205 * sf_risk_percent)

    # -------------------------------------------------------------------------
    # Logic Results (Display)
    # -------------------------------------------------------------------------
    res_sf_text, res_sf_color = "Wait...", (100, 100, 100)
    if avg_max_foot > 0:
        if avg_max_shoulder <= (avg_max_foot / 2): 
            res_sf_text, res_sf_color = f"S&F: OK (% Risk: {sf_risk_percent:.1f})", color_ok
        else: 
            res_sf_text, res_sf_color = f"S&F: Risk (% Risk: {sf_risk_percent:.1f})", color_risk
    
    res_knee_text, res_knee_color = "Knee: Wait...", (100, 100, 100)
    if avg_max_knee > 0:
        if avg_max_knee > 140: 
            res_knee_text, res_knee_color = f"Knee: OK (% Risk: {knee_risk_percent:.1f})", color_ok
        else: 
            res_knee_text, res_knee_color = f"Knee: Risk (% Risk: {knee_risk_percent:.1f})", color_risk

    res_fh_text, res_fh_color = "Ft.Height: Wait...", (100, 100, 100)
    if avg_max_fh != 0:
        if avg_max_fh > 2: 
            res_fh_color = color_ok; main_status = "OK"
        else: 
            res_fh_color = color_risk; main_status = "Risk"
        res_fh_text = f"Ft.Height: {main_status} (% Risk: {fh_status_str})"

    # วาดผลลัพธ์
    cv2.putText(final_image, res_sf_text, (start_x + 10, box_y_start + 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, res_sf_color, 2)
    cv2.putText(final_image, res_knee_text, (start_x + 10, box_y_start + 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, res_knee_color, 2)
    cv2.putText(final_image, res_fh_text, (start_x + 10, box_y_start + 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, res_fh_color, 2)
    
    cv2.putText(final_image, f"% Total Risk: {total_risk:.1f}", (start_x + 10, box_y_start + 160), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)

    last_final_image = final_image
    cv2.imshow('Side-by-Side Analysis', final_image)

    key = cv2.waitKey(30) 
    if key == ord('q') or key == ord('Q') or key == 27: 
        break

cap.release()
cv2.destroyAllWindows()