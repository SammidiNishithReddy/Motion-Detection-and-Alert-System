import cv2
import numpy as np
import smtplib
from email.message import EmailMessage
import mimetypes
import os

def send_email_alert(image_path):
    """
    Sends an email alert with the captured image as an attachment.
    Args:
    - image_path (str): Path to the image to be sent.
    """
    # Your email credentials
    sender_email = "sammidinishithreddy@gmail.com"
    receiver_email = "sammidinishithreddy@gmail.com"
    password = "yvqm gnoq pfdd jqfm"

    # Set up the email message
    msg = EmailMessage()
    msg['Subject'] = "Motion Detected!"
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg.set_content('Motion has been detected by the security camera.')

    # Read and attach the image
    with open(image_path, 'rb') as img_file:
        img_data = img_file.read()
        img_type = mimetypes.guess_type(img_file.name)[0].split('/')[1]
        img_name = os.path.basename(img_file.name)

    msg.add_attachment(img_data, maintype='image', subtype=img_type, filename=img_name)

    # Send the email
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(sender_email, password)
        smtp.send_message(msg)
        print("Email sent with motion detected image.")

def detect_motion():
    """
    Detects motion using the webcam and captures one image when motion is detected.
    Sends an email alert with the captured image, then closes the camera.
    """
    video_capture = cv2.VideoCapture(0)
    background_frame = None
    contour_area_threshold = 4000
    motion_persistence = 5
    motion_frame_count = 0

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        #Applies Gaussian blur to reduce noise and improve motion detection accuracy.
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
        #first loop iteration
        if background_frame is None:
            background_frame = gray_frame
            continue

        frame_diff = cv2.absdiff(background_frame, gray_frame)
        _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) < contour_area_threshold:
                continue

            motion_detected = True
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Security Camera", frame)

        if motion_detected:
            motion_frame_count += 1
        else:
            motion_frame_count = 0

        if motion_frame_count >= motion_persistence:
            img_name = "motion_detected.png"
            cv2.imwrite(img_name, frame)
            print(f"Motion detected! Image saved as {img_name}.")
            
            send_email_alert(img_name)
            break  # Exit the loop after capturing and sending the email

        background_frame = gray_frame

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close all windows
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_motion()
