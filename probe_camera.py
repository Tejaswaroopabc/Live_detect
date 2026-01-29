# probe_camera.py
import cv2

def find_cameras(max_idx=8):
    available = []
    for i in range(max_idx + 1):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # use CAP_DSHOW on Windows
        if cap is not None and cap.isOpened():
            available.append(i)
            cap.release()
    return available

if __name__ == "__main__":
    cams = find_cameras(8)
    if cams:
        print("Available camera indices:", cams)
    else:
        print("No cameras detected. Try replugging the USB camera or check drivers.")
