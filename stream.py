import cv2
import numpy as np
import time as tm
import matplotlib as plt
import pickle

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
#cap = cv2.VideoCapture('logo_wentzl_cam_be5c6c.stream-11_30-2017-12fps.mp4.mp4')
fileNames = [
'logo_wentzl_cam_be5c6c.stream-11_30-2017-12fps.mp4.mp4', 'logo_wentzl_cam_be5c6c.stream-12_17-2017-12fps.mp4.mp4'
]

for videoName in fileNames:
    cap = cv2.VideoCapture(videoName)

    # Check if camera opened successfully
    if (cap.isOpened()== False):
      print("Error opening video stream or file")

    def drawRectangle(frame ,cord = (0,0,0,0), color=(255,0,0)):
        cv2.line(frame, (cord[0], cord[1]), (cord[0] + cord[2], cord[1]), (255,0,0), 1)
        cv2.line(frame, (cord[0], cord[1]), (cord[0], cord[1] + cord[3]), (255,0,0), 1)
        cv2.line(frame, (cord[0] + cord[2], cord[1]), (cord[0]+cord[2], cord[1] + cord[3]), (255,0,0), 1)
        cv2.line(frame, (cord[0], cord[1]+cord[3]), (cord[0]+cord[2], cord[1]+cord[3]), (255,0,0), 1)

    def calculateCannyRatio(cannyImg):
        whites = 0
        blacks = 0
        for i in range(cannyImg.shape[0]):
            for j in range(cannyImg.shape[1]):
                if cannyImg[i, j] > 200:
                    whites += 1
                else:
                    blacks += 1

        return whites/blacks


    # Read until video is completed
    wojciechData = []
    mariackiData = []
    frameData = []
    frames = 0
    mariackiFrames = []
    wojciechFrames = []

    while(cap.isOpened()):
      # Capture frame-by-frame
      ret, frame = cap.read()
      if ret == True:

        #cv2.line(frame, (700,200), (700,300), (255,0,0), 5)
        #cv2.line(frame, (700,200), (700,300), (255,0,0), 5)

        # Display the resulting frame

        mariacki = frame[100:200, 630:730]
        wojciech = frame[450:550, 1010:1110]

        # mariacki = mariacki * (1/255)
        # wojciech = wojciech * (1/255)
        
        # exit()
        tempWojciech = cv2.resize(cv2.cvtColor(wojciech, cv2.COLOR_BGR2GRAY), (50, 50), interpolation=cv2.INTER_AREA)
        tempMariacki = cv2.resize(cv2.cvtColor(mariacki, cv2.COLOR_BGR2GRAY), (50, 50), interpolation=cv2.INTER_AREA)
        wojToApp = np.array([])
        marToApp = np.array([])
        # print(tempWojciech[0], tempWojciech[1])
        for i in range(50):
            wojToApp = np.concatenate((wojToApp, tempWojciech[i]), axis=0)
            marToApp = np.concatenate((marToApp, tempMariacki[i]), axis=0)

        mariackiFrames.append(wojToApp)
        wojciechFrames.append(marToApp)

        #normalize data
        mariackiFrames[-1] = mariackiFrames[-1] * (1/255)
        wojciechFrames[-1] = wojciechFrames[-1] * (1/255)


        # wojciechFrames.append(cv2.resize(cv2.cvtColor(wojciech, cv2.COLOR_BGR2GRAY), (50, 50), interpolation=cv2.INTER_AREA))
        # print(mariackiFrames[0][0])
        # break

        #cv2.imshow('Mariacki', mariacki)
        #cv2.imshow("Wojciech", wojciech)

        cannyMariacki = cv2.Canny(mariacki, 50, 50)
        cannyWojciech = cv2.Canny(wojciech, 50, 50)
        #cannyFrame = cv2.Canny(frame, 50, 50)

        #drawRectangle(frame, (630, 100, 100, 100))
        #drawRectangle(frame, (1010,450, 100,100))

        #cv2.imshow('Original Image',frame)
        mariackiData.append(calculateCannyRatio(cannyMariacki))
        wojciechData.append(calculateCannyRatio(cannyWojciech))

        #frameData.append(calculateCannyRatio(cannyFrame))

        print(frames,": ", wojciechData[-1], mariackiData[-1])
        #cv2.imshow("MariackiLaplacian" ,cv2.Canny(cv2.Laplacian(mariacki, cv2.CV_8U), 50, 50))
        #cv2.imshow("WojciechLaplacian" ,cv2.Canny(cv2.Laplacian(wojciech, cv2.CV_8U), 50, 50))
        frames += 1

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
          break


      # Break the loop
      else:
        break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()
    tempData = []
    temp = [mariackiData, wojciechData]
    # print(temp)
    try:
        tempData = (pickle.load(open("data.pickle", "rb")))
        # print(tempData)
        temp = [temp[0] + tempData[0], temp[1] + tempData[1]]
        # print("\n\n\n\n")
        # print(temp)
    except:
        pass

    pickle.dump(temp, open("data.pickle", "wb"))
    try:
        tempW = pickle.load(open("wojciechFrames.pickle", "rb"))
    except:
        tempW = []

    pickle.dump(tempW + wojciechFrames, open("wojciechFrames.pickle", "wb"))

    try:
        tempM = pickle.load(open("mariackiFrames.pickle", "rb"))
    except:
        tempM = []
    pickle.dump(tempM + mariackiFrames, open("mariackiFrames.pickle", "wb"))
    print("dumping data finished!")