
from queue import Queue
import numpy as np
import cv2

vid = cv2.VideoCapture(0)

frames_3=np.zeros((3,480,640))

ret, frame = vid.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
frames_3 = np.insert(frames_3, frames_3.shape[0], gray, axis=0)
# print(frames_3.shape)

frames_3 = np.delete(frames_3, 0, axis=0)
print(frames_3)
print(frames_3.shape)

ret, frame = vid.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
frames_3 = np.insert(frames_3, frames_3.shape[0], gray, axis=0)
# print(frames_3.shape)

frames_3 = np.delete(frames_3, 0, axis=0)
print(frames_3)
print(frames_3.shape)

ret, frame = vid.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
frames_3 = np.insert(frames_3, frames_3.shape[0], gray, axis=0)
# print(frames_3.shape)

frames_3 = np.delete(frames_3, 0, axis=0)
print(frames_3)
print(frames_3.shape)

image3 = cv2.normalize(frames_3[2], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
detector=cv2.xfeatures2d.SIFT_create()
kp,Desc=detector.detectAndCompute(image3,None)
freakExtractor = cv2.xfeatures2d.FREAK_create()
trainKP,trainDesc = freakExtractor.compute(image3, kp)
# trainKP,trainDesc=detector.detectAndCompute(image3,None)


fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
#creating an object of videoWriter for writing video
out = cv2.VideoWriter('freak_video.avi',fourcc, 5, (int(vid.get(3)),int(vid.get(4))))
while (True):

    # Capture the video frame
    # by frame
    # for i in range(3):
    #     vid.grab()
    ret, frame = vid.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print(gray.shape)
    frames_3 = np.insert(frames_3, frames_3.shape[0], gray, axis=0)
    # print(frames_3.shape)

    frames_3 = np.delete(frames_3, 0, axis=0)
    print(frames_3)
    print(frames_3.shape)

    image1 = cv2.normalize(frames_3[0], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    image2 = cv2.normalize(frames_3[1], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    image3 = cv2.normalize(frames_3[2], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    sift = cv2.xfeatures2d.SIFT_create()
    freakExtractor = cv2.xfeatures2d.FREAK_create()
    kp3, Desc3 = sift.detectAndCompute(image3, None)
    kp1, Desc1 = sift.detectAndCompute(image1, None)
    kp2, Desc2 = sift.detectAndCompute(image2, None)


    keypoints1, descriptors1 = freakExtractor.compute(image1, kp1)
    keypoints2, descriptors2 = freakExtractor.compute(image2, kp2)
    keypoints3, descriptors3 = freakExtractor.compute(image3, kp3)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)
    # # match descriptors of both images
    matches = bf.match(descriptors1, descriptors2)
    # matches = sorted(matches, key=lambda x: x.distance)
    # print("match",matches)

    points1=[]
    points2=[]
    counter=0
    goodmatches=[]
    for match in matches:

        if(match.distance < 50) :
            goodmatches.append(match)
        else:
            goodmatches.append('nan')

    print(len(matches))
    print("good",len(goodmatches))
    print("train",len(trainKP))
    print("3",len(keypoints3))

    for match in goodmatches:

        if match!='nan':
            print(match.queryIdx)

            point1 = keypoints1[match.queryIdx].pt
            points1.append(point1)

            # if (len(keypoints2) < match.queryIdx -1):

            point2 = keypoints2[match.trainIdx].pt
            points2.append(point2)



    print(len(points2))
    print(len(points1))
    x_1=0
    y_1=0
    x_2=0
    y_2=0
    for i in range(len(points1)):
        x_1 = x_1 + points1[i][0]
        y_1 = y_1 + points1[i][1]
    x_mean = x_1/len(points1)
    y_mean = y_1/len(points1)

    for i in range(len(points2)):
        x_2 = x_2 + points2[i][0]
        y_2 = y_2 + points2[i][1]
    x_mean_2 = x_2/len(points2)
    y_mean_2 = y_2/len(points2)


    # for i in range(len(points1)):
    #     cv2.arrowedLine(frame, (np.int(points1[i][0]),np.int(points1[i][1])), (np.int(points2[i][0]),np.int(points2[i][1])), (0, 0, 255),thickness=3)

    cv2.arrowedLine(frame, (np.int(x_mean), np.int(y_mean)), (np.int(x_mean_2), np.int(y_mean_2)),(0, 0, 255),thickness=3)
######################################
    # matches_2 = bf.match(descriptors2, descriptors3)
    matches = bf.match(descriptors2, descriptors3)
    # matches = sorted(matches, key=lambda x: x.distance)
    # print("match",matches)

    points1 = []
    points2 = []
    counter = 0
    goodmatches = []
    for match in matches:

        if (match.distance < 50):
            goodmatches.append(match)
        else:
            goodmatches.append('nan')

    print(len(matches))
    print("good", len(goodmatches))
    print("train", len(trainKP))
    print("3", len(keypoints3))

    for match in goodmatches:

        if match != 'nan':
            print(match.queryIdx)

            point1 = keypoints2[match.queryIdx].pt
            points1.append(point1)

            # if (len(keypoints2) < match.queryIdx -1):

            point2 = keypoints3[match.trainIdx].pt
            points2.append(point2)

    print(len(points2))
    print(len(points1))
    x_1 = 0
    y_1 = 0
    x_2 = 0
    y_2 = 0
    for i in range(len(points1)):
        x_1 = x_1 + points1[i][0]
        y_1 = y_1 + points1[i][1]
    x_mean = x_1 / len(points1)
    y_mean = y_1 / len(points1)

    for i in range(len(points2)):
        x_2 = x_2 + points2[i][0]
        y_2 = y_2 + points2[i][1]
    x_mean_2 = x_2 / len(points2)
    y_mean_2 = y_2 / len(points2)

    # for i in range(len(points1)):
    #     cv2.arrowedLine(frame, (np.int(points1[i][0]), np.int(points1[i][1])),(np.int(points2[i][0]), np.int(points2[i][1])), (0, 255, 0), thickness=3)
    cv2.arrowedLine(frame, (np.int(x_mean), np.int(y_mean)), (np.int(x_mean_2), np.int(y_mean_2)), (0, 255, 0),
                    thickness=3)

    out.write(frame)
    cv2.imshow('image', frame)

    if ret == False:
        break
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

vid.release()
out.release()
    #
    #
    # freakExtractor = cv2.xfeatures2d.FREAK_create()
    # keypoints, descriptors = freakExtractor.compute(gray, kp)


