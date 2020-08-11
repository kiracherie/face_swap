"""
This project takes video from the computer's camera, finds the faces in the camera, then if two faces are found
it will do a face swap!
"""

# load libraries to be used
import numpy as np
import cv2

""" Resize image to a new shape """
def resize_image(image, shape):
    # old width and height from original image
    old_height, old_width = image.shape[:2]
    new_height, new_width = shape

    # find dcale factor to multiply original frame's width and height by
    scale = min(new_width/old_width, new_height/old_height)

    # Final size of frame scaled to fit the width x height
    new_height = int(old_height * scale)
    new_width = int(old_width * scale)

    new_scale = (new_width, new_height)

    return cv2.resize(image, new_scale)

""" create a mask with an ellipse the same size as the given image """
def circle_mask(image):
    r, c = image.shape[:2]
    center = (c//2, r//2)
    x = int((c*.7) // 2)
    y = int(r // 2)
    mask = np.zeros((r,c), dtype=np.uint8)

    # create white circle
    mask = cv2.ellipse(mask, center, (x, y), 0, 0, 360, (255, 255, 255), -1)
    return mask

""" finds the frontal faces in a given image """
def find_objects(image, cascade):
    # gray copy of the background image
    background_image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    background_image_gray = cv2.equalizeHist(background_image_gray)

    # find face in the background image
    faces_rects = cascade.detectMultiScale(background_image_gray, scaleFactor = 1.2, minNeighbors = 5)

    return faces_rects

""" takes the rectangle of the face in an image and crops the image to include just the face """
def crop_image(face_rect, image):
    print("cropping image")
    # point and size of the rectangle
    x,y,w,h = face_rect

    # Returns cropped image with just the face
    cropped_image = np.zeros((w, h, 3), dtype=np.uint8)
    cropped_image = image[y:y+h, x:x+w]

    return cropped_image

""" 
Finds the faces in the background image, resizes the object image and mask to the size of those faces
then pastes the object image onto the faces in the background image.
"""
def apply_face_swap(face_rects, final_image):
    # create images of just the faces and create masks for them
    face1 = crop_image(face_rects[0], final_image)
    face2 = crop_image(face_rects[1], final_image)

    face1_mask = circle_mask(face1)
    face2_mask = circle_mask(face2)

    # store the face size and centers
    face1_size = face1.shape[:2]
    face2_size = face2.shape[:2]

    # finding the centers of face1 and face2
    x1,y1,w1,h1 = face_rects[0]
    face1_center = (((x1+w1 // 2), (y1+h1 //2)))
    x2,y2,w2,h2 = face_rects[1]
    face2_center = (((x2+w2 // 2), (y2+h2 //2)))

    # resize face and masks to fit the other's face
    face1_resized = resize_image(face1, face2_size)
    face1_mask_resized = resize_image(face1_mask, face2_size)

    face2_resized = resize_image(face2, face1_size)
    face2_mask_resized = resize_image(face2_mask, face1_size)

    # apply seamless clone for both faces (the actual face swap)
    final_image = cv2.seamlessClone(face2_resized, final_image, face2_mask_resized, face1_center, cv2.NORMAL_CLONE)
    final_image = cv2.seamlessClone(face1_resized, final_image, face1_mask_resized, face2_center, cv2.NORMAL_CLONE)

    return final_image


""" Main: Finding faces in an image then applying a face swap if 2 faces are found """
def main2():
    haar_cascade_face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    cap = cv2.VideoCapture(0) # Notice the '0' instead of a filename
    #loop will only break if 'q' key is pressed
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # find the faces in the frame
        faces = find_objects(frame, haar_cascade_face)

        num_faces = len(faces)  
        print("Found {0} faces!".format(num_faces))
        
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # apply face swap if two faces are found
        if (num_faces == 2):
            frame = apply_face_swap(faces, frame)        
            
        # Display the resulting frame
        cv2.imshow('frame', frame)
        
        # Wait for 'q' to quit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

# -------------------------------------------------------------------------------------------
main2()
