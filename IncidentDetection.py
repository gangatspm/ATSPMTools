import base64
import json
from collections import defaultdict
import cv2
import numpy as np
import smtplib
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from ultralytics import YOLO

# Email setup
def send_email_alert(vehicle_id, last_email_time, sender, receiver, pwd, frame):
    sender_email = sender
    receiver_email = receiver
    password = pwd
    
    subject = f"Alert: Vehicle {vehicle_id} stationary for over 1 minute"
    # Convert the frame to a base64 string
    _, buffer = cv2.imencode('.png', frame)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Create an HTML email body with the embedded image
    body = f"""
    <html>
        <body>
            <p>Vehicle with ID {vehicle_id} has been stationary for over 1 minute.</p>
            <p>Here is the video frame:</p>
            <img src="data:image/png;base64,{image_base64}">
        </body>
    </html>
    """
    
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = ", ".join(receiver_email)  # Join multiple emails with commas
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'html'))  # Set email content to HTML
    
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, msg.as_string())  # Pass list directly
        print(f"Alert email with embedded frame sent for vehicle {vehicle_id}")
        last_email_time = time.time()  # Update last email sent time
    except Exception as e:
        print(f"Failed to send email: {e}")
    
    return last_email_time

# Main tracking function
def track_vehicles(video_path, stationary_threshold, email_interval, skip_frames):
    # Load the YOLO model
    model = YOLO("yolo11n.pt")

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    #rtsp://10.202.3.82/stream1
    #rtsp://10.205.3.35:554/VideoInput/1/mpeg4/1

    # Store the track history and stationary timer
    track_history = defaultdict(lambda: [])
    stationary_timer = {}
    last_email_time = 0  # Timestamp of the last email sent

    # Frame skipping variable
    frame_count = 0  # Frame counter

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        frame_count += 1

        # Skip frames based on the skip_frames value
        if frame_count % skip_frames != 0:
            continue

        if success:
            # Run YOLO tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True, verbose=False)

            if results[0].boxes.id is None:
                continue

            # # Get the boxes and track IDs
            # boxes = results[0].boxes.xywh.cpu()
            # track_ids = results[0].boxes.id.int().cpu().tolist()

            # Filter detections by class (car, motorcycle, truck)
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            valid_classes = [2, 3, 5, 7]  # car, motorcycle, truck
            valid_detections = [i for i, cls_id in enumerate(class_ids) if cls_id in valid_classes]

            if not valid_detections:
                continue

            # Get the valid boxes and track IDs (filter by valid detections)
            boxes = results[0].boxes.xywh.cpu()[valid_detections]  # Filter boxes
            track_ids = results[0].boxes.id.cpu()[valid_detections].int().tolist()  # Filter IDs and convert to list


            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Get the current time for stationary tracking
            current_time = time.time()
            any_stationary_vehicles = False

            # Plot the tracks and check for stationary vehicles
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                if len(track) > 30:  # retain only 30 track points for each object
                    track.pop(0)

                # Update stationary timer
                if track_id in stationary_timer:
                    # Check if the vehicle has been stationary for over the threshold
                    if current_time - stationary_timer[track_id] > stationary_threshold:
                        any_stationary_vehicles = True
                        if current_time - last_email_time > email_interval:
                            last_email_time = send_email_alert(track_id, last_email_time, config["sender_email"], config["receiver_email"], config["password"], annotated_frame)
                else:
                    # Start the stationary timer for a new vehicle
                    stationary_timer[track_id] = current_time

                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=5)

            # Remove vehicle IDs no longer detected
            vehicle_ids_in_frame = set(track_ids)
            stationary_timer = {vid: timer for vid, timer in stationary_timer.items() if vid in vehicle_ids_in_frame}

            # Display the annotated frame
            cv2.imshow("Incident Detection Using YOLO11 Tracking", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Load configuration from a JSON file
    with open("config.json", "r") as f:
        config = json.load(f)

    # Call the main function with the values from the config file
    track_vehicles(config["video_path"], config["stationary_threshold"], config["email_interval"], config["skip_frames"])
