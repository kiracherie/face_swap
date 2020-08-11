"""
This project will be able to recognize faces, track them, then when two faces appear on the screen
a face swap will occur.

In this version of the project you are able to detect a cat face in an object picture file then
find faces in a background picture file and replace them with the cat face.

HUMAN_FACE_IMAGE: replace with the directory and filename of the image with faces
    you want to replace with cats
CAT_FACE_IMAGE: image with a cat face to replace human faces with

Note: human and cat faces must be of good quality and be facing the camera to work
"""

# load libraries to be used
import numpy as np
import cv2

HUMAN_FACE_IMAGE = 'pic_samples/photo_of_face.jpg'
CAT_FACE_IMAGE = 'pic_samples/cat_face.jpg'
SAVE_IMAGE = 'final_images/final_image.png'

""" Resize image to new shape """
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

""" 
Finds the faces in the background image, resizes the object image and mask to the size of those faces
then pastes the object image onto the faces in the background image.
"""
def apply_face_object(object_image, object_mask, background_image):
    # gray copy of the background image
    background_image_gray = cv2.cvtColor(background_image, cv2.COLOR_BGR2GRAY)
    background_image_gray = cv2.equalizeHist(background_image_gray)
    print("gray image created")

    # final_image is copy of background
    final_image = background_image.copy()

    # find face in the background image
    haar_cascade_face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces_rects = haar_cascade_face.detectMultiScale(background_image_gray, scaleFactor = 1.2, minNeighbors = 5)

    # Let us print the no. of faces found
    print('Faces found: ', len(faces_rects))

    # find the center of the face in the image then past the object image over that point
    for (x,y,w,h) in faces_rects:
        center = ((x+w // 2), (y+h //2))
        shape = (int(h * 1.2), int(w * 1.3))                        # shape the object should be to cover the face
        print("center of face:", center)
        print("shape of the face:", shape)

        # resize mask
        object_image_resized = resize_image(object_image, shape)
        cv2.imshow('object image', object_image_resized)
        object_mask_resized = resize_image(object_mask, shape)

        # apply seamless clone
        final_image = cv2.seamlessClone(object_image_resized, final_image, object_mask_resized, center, cv2.NORMAL_CLONE)

        # # draw outline of face
        # final_image = cv2.ellipse(final_image, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)

    return final_image

def find_catface(cat_image):
    gray_image = cv2.cvtColor(cat_image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.equalizeHist(gray_image)

    haar_cascade_face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalcatface_extended.xml")

    final_image = cat_image.copy()

    cat_rects = haar_cascade_face.detectMultiScale(gray_image, scaleFactor = 1.2, minNeighbors = 5)

    # Let us print the no. of faces found
    num_cats = len(cat_rects)
    print('Cats found: ', num_cats)

    # error checking
    if (num_cats > 1) or (num_cats == 0):
        print("Number of cats found:", num_cats)
        print("Retry with one cat face")
        return 0, 0

    # find the center of the cat
    x,y,w,h = cat_rects[0]
    center = ((x+w // 2), (y+h //2))
    shape = (h, w)
    print("center of cat:", center)
    print("shape of the cat:", shape)

    # draw outline
    final_image = cv2.ellipse(final_image, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)

    # Returns cropped image with just the cat face
    cropped_cat_image = np.zeros((w, h, 3), dtype=np.uint8)
    cropped_cat_image = cat_image[y:y+h, x:x+w]

    return 1, cropped_cat_image

    # # show final image
    # cv2.imshow('found cat', final_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows

def circle_mask(image):
    r, c = image.shape[:2]
    center = (c//2, r//2)
    x = int((c*.7) // 2)
    y = int(r // 2)
    mask = np.zeros((r,c), dtype=np.uint8)

    #create white circle
    mask = cv2.ellipse(mask, center, (x, y), 0, 0, 360, (255, 255, 255), -1)
    return mask

""" Main: Finding a cat face and pasting it onto people's faces """
def main2():
    object_filename = CAT_FACE_IMAGE
    background_filename = HUMAN_FACE_IMAGE

    background_image = cv2.imread(background_filename)
    object_image = cv2.imread(object_filename)
    error_check, object_image = find_catface(object_image)
    print()
    if (error_check == 0):
        print("Program failed")
        return

    object_mask = circle_mask(object_image)

    final_image = apply_face_object(object_image, object_mask, background_image)
    print("applied mask")

    # saving image
    cv2.imwrite(SAVE_IMAGE, final_image)


# -------------------------------------------------------------------------------------------
main2()
