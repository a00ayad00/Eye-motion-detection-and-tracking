import cv2

input_path = r'D:\Projects\Computer Vision\Eye motion detection and tracking\eye.mp4'

cap = cv2.VideoCapture(input_path)


w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"X264")
output_path = r'D:\Projects\Computer Vision\Eye motion detection and tracking\output.mp4'
out = cv2.VideoWriter(output_path, fourcc, 30, (w, 426))

while True:
    done, frame = cap.read()
    
    if done is False: break
        
    roi = frame[369: 795, :]

    rows, cols, _ = roi.shape
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    _, threshold = cv2.threshold(blur, 30, 250, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)

        # cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 3)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.line(roi, (x + int(w/2), 0), (x + int(w/2), rows),
                 (0, 255, 0), 2)
        cv2.line(roi, (0, y + int(h/2)), (cols, y + int(h/2)),
                 (0, 255, 0), 2)
        
        
        break
        

    # cv2.imshow("Threshold", threshold)
    # cv2.imshow("gray", gray)
    cv2.imshow("original Image", roi)
    
    out.write(roi)
    
    key = cv2.waitKey(30)
    if key == ord("a"):
        break

cv2.destroyAllWindows()
cap.release()
out.release()