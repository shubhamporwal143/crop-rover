import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas.to('cpu')
midas.eval()
transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.small_transform
url="http://192.168.123.249:8080/video"
cap = cv2.VideoCapture("video4.mp4")
while cap.isOpened(): 
    ret, frame = cap.read()
    cv2.imshow('TH121',cv2.resize(frame, (640, 480)))
#     height,width =img.shape
#     img=img[height//2:height,width//4: 3*width//4]
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #cv2.imshow("SC",img)
    imgbatch = transform(img).to('cpu')
    with torch.no_grad(): 
        prediction = midas(imgbatch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size = img.shape[:2], 
            mode='bicubic', 
            align_corners=False
        ).squeeze()
        output = prediction.cpu().numpy()
        img=cv2.normalize(output, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
        img = (img*255).astype(np.uint8)
#         img = cv2.applyColorMap(img , cv2.COLORMAP_PINK )
    cv2.imshow('CV2Frame', cv2.resize(img, (640, 480)))
    width,height =img.shape
    print(width,height)
#     for mob cam
#     img=img[0:850, 0:1920 
# for lap cam
    img=img[0:350,0:640]
            
    _, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    contours, hierrachry = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objects_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:
            
            objects_contours.append(cnt)
    contours = objects_contours

    # Draw objects boundaries
    for cnt in contours:

        # select minimum rect_paramenter
        rect = cv2.minAreaRect(cnt)
        #Center Width/length Angle
        (x, y), (w, h), angle = rect
        print(rect)

        # pixel to cm
        #w = w*0.0264583333
        #h=h*0.0264583333

        # show rectangle
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # center point of image
        cv2.circle(img, (int(x-(w//2)), int(y)), 50, (0, 0, 255), -1)

        # draw polygon
        cv2.polylines(img, [box], True, ( 255,0,0), 2)

        # show width/height
        #cv2.putText(img, "s".format(round((x-(w//2)), 1)), (int(x - 100), int(y - 20)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
#         cv2.putText(img, "Height {} cm".format(round(h, 1)), (int(x - 100), int(y + 20)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
        #cv2.putText(img, "obstacle", (int(x - 100), int(y - 20)),cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)  
    cv2.imshow('TH1',cv2.resize(img, (640, 480)))
    plt.pause(0.00001)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        cap.release()
        cv2.destroyAllWindows()

plt.show()