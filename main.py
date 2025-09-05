import cv2

eye = cv2.CascadeClassifier('haarcascade_eye.xml')
face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile = cv2.CascadeClassifier('haarcascade_smile.xml')

cap = cv2.VideoCapture(0)

while True:
    ret , frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face.detectMultiScale(gray, 1.1, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye.detectMultiScale(roi_gray,1.1, 10)
        if len(eyes) > 0:
            cv2.putText(frame, "eyes detected ", (x,y-30), cv2.FONT_HERSHEY_SIMPLEX , 0.6 ,(0, 255, 0), 2)

        smiles = smile.detectMultiScale(roi_gray,1.7, 20)
        if len(smiles) > 0:
            cv2.putText(frame, "smile detected ", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX , 0.6 ,(0, 255, 0), 2)
        
    cv2.imshow('smart face detection', frame)
    
    if cv2.waitKey(1) & 0XFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()