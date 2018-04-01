#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import argparse
import copy

def frame_resize(frame, n=2):
    return cv2.resize(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)))

if __name__ == '__main__':

    # KCF
    tracker_kcf = cv2.TrackerKCF_create()

    # TLD
    tracker_tld = cv2.TrackerTLD_create()

    # MedianFlow
    tracker_mf = cv2.TrackerMedianFlow_create()

     # setup some arguments
    parser = argparse.ArgumentParser(description='yolov2_darknet_predict for video')
    parser.add_argument('--video_file', '-v', type=str, default=False,help='path to video')
    parser.add_argument('--camera_ID', '-c', type=int, default=0,help='camera ID')
    parser.add_argument('--save_name', '-s', type=str, default=False,help='camera ID')
    args = parser.parse_args()

    # prepare to get image frames
    if not args.video_file == False:
        cap = cv2.VideoCapture(args.video_file)
    else:
        cap = cv2.VideoCapture(args.camera_ID)

    # cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture("3.avi")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = frame_resize(frame)
        bbox = (0,0,10,10)
        bbox = cv2.selectROI(frame, False)
        ok_kcf = tracker_kcf.init(frame, bbox)
        ok_tld = tracker_tld.init(frame, bbox)
        ok_mf = tracker_mf.init(frame, bbox)
        # cv2.destroyAllWindows()
        break

    while True:
        ret, frame = cap.read()
        frame = frame_resize(frame)
        if not ret:
            k = cv2.waitKey(1)
            if k == 27 :
                break
            continue

        # Start timer
        # timer = cv2.getTickCount()

        track_kcf, bbox_kcf = tracker_kcf.update(frame)
        track_tld, bbox_tld = tracker_tld.update(frame)
        track_mf, bbox_mf = tracker_mf.update(frame)

        # fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        if track_kcf:
            # Tracking success
            p1_kcf = (int(bbox_kcf[0]), int(bbox_kcf[1]))
            p2_kcf = (int(bbox_kcf[0] + bbox_kcf[2]), int(bbox_kcf[1] + bbox_kcf[3]))
            cv2.rectangle(frame, p1_kcf, p2_kcf, (0,255,0), 2, 1)
            cv2.putText(frame, "KCF", (p1_kcf[0]+10,p1_kcf[1]+25),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 1, cv2.LINE_AA);
        else :
            cv2.putText(frame, "KCF Failure", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA);

        if track_tld:
            # Tracking success
            p1_tld = (int(bbox_tld[0]), int(bbox_tld[1]))
            p2_tld = (int(bbox_tld[0] + bbox_tld[2]), int(bbox_tld[1] + bbox_tld[3]))
            cv2.rectangle(frame, p1_tld, p2_tld, (255,0,0), 2, 1)
            cv2.putText(frame, "TLD", (p1_tld[0]+10,p1_tld[1]+25),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1, cv2.LINE_AA);
        else :
            cv2.putText(frame, "TLD Failure", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA);

        if track_mf:
            # Tracking success
            p1_mf = (int(bbox_mf[0]), int(bbox_mf[1]))
            p2_mf = (int(bbox_mf[0] + bbox_mf[2]), int(bbox_mf[1] + bbox_mf[3]))
            cv2.rectangle(frame, p1_mf, p2_mf, (0,0,255), 2, 1)
            cv2.putText(frame, "Median Flow", (p1_mf[0]+10,p1_mf[1]+25),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, cv2.LINE_AA);
        else :
            cv2.putText(frame, "Median Flow Failure", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA);



        # cv2.putText(frame, "FPS : " + str(int(fps)), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA);

        cv2.imshow("Tracking", frame)

        k = cv2.waitKey(1)
        if k == 27 :
            break

cap.release()
cv2.destroyAllWindows()
