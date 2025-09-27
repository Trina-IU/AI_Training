import cv2
import numpy as np

p = "videos/amoxicillin.mp4"
cap = cv2.VideoCapture(p)
print('video_path=', p)
print('isOpened=', cap.isOpened())
fps = cap.get(cv2.CAP_PROP_FPS)
print('fps=', fps)
i = 0
maxf = 10
while i < maxf:
    ret, frame = cap.read()
    print('frame', i, 'ret=', ret)
    if not ret:
        break
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        print('cvtColor error:', e)
        break
    h, w = gray.shape
    print('shape=', gray.shape)
    scale = min(128.0 / w if w > 0 else 1.0, 32.0 / h if h > 0 else 1.0)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = 255 * np.ones((32, 128), dtype=np.uint8)
    xo = (128 - nw) // 2
    yo = (32 - nh) // 2
    canvas[yo:yo+nh, xo:xo+nw] = resized
    _, th = cv2.threshold(canvas, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    sharp = float(cv2.Laplacian(th, cv2.CV_64F).var())
    ink = float((th < 128).mean())
    print('sharp=', sharp, 'ink=', ink)
    i += 1
cap.release()

print('diag done')
