import cv2
import IPython.display as display
from IPython.display import clear_output
import time

# Function to play video
def play_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(fps)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to RGB (OpenCV uses BGR by default)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Clear the previous frame
        clear_output(wait=True)
        
        # Display frame
        display.display(display.Image(data=cv2.imencode('.jpg', frame_rgb)[1].tobytes()))
        
        # Delay to match video frame rate
        time.sleep(1/30)  # Adjust the delay based on the frame rate of your video
    
    cap.release()

# Example usage
video_path = 'test_videos/IRON MAN 2 (2010) .mp4'
play_video(video_path)
