import cv2

# Create a callback function to handle mouse events
def get_mouse_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Coordinates: x={x}, y={y}")

# Create a video capture object
cap = cv2.VideoCapture('footage.mp4')  # Replace 'your_video.mp4' with your video file
cap.set(3,1280)
cap.set(4,720)
# Set a window to display the video
cv2.namedWindow('Video')

# Set the mouse callback function
cv2.setMouseCallback('Video', get_mouse_coordinates)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Display the frame in the window
    cv2.imshow('Video', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

# Release the video capture object and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
