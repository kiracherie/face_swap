"""
This project takes an image from the computer's hard drive, finds the faces in the image, then swaps the faces if two faces are found!

Version 2:
This version uses code adapted from pysource's face swap method using Delaunay triangulation
https://pysource.com/2019/05/28/face-swapping-explained-in-8-steps-opencv-with-python/

To do the face swap, make sure to install the dlib library on your computer, and to store the shape_predictor_68_face_landmarks.dat file
somewhere on your computer.

FILENAME: the image file that you want to apply faceswap to
SAVE_IMAGE: the file name that you want to save the resulting face swap to
PREDICTOR_FILE: the file directory that you saved 'shape_predictor_68_face_landmarks.dat' to

Issues:
Image to be face swapped has to be of good quality in order for the program to find faces.
The final image contains some black artifacts from the face swap.
Some people's faces get wonky, especially those with glasses and a turned head.
Things that are touching the face will become unnatural artifacts in the resulting face swap.
"""

import sys           #to abort program
import cv2
import numpy as np
import dlib

# Change the constants below
FILENAME = 'two_people11.jpg'
SAVE_IMAGE = 'final_image.png'
PREDICTOR_FILE = '/Users/Kira/Documents/BYUI/2020 Winter/CS312 Vision and Graphics/face-alignment/shape_predictor_68_face_landmarks.dat'

""" helper functions """
def crop_image(face_rect, image):
    print("cropping image")
    # point and size of the rectangle
    x,y,w,h = face_rect

    # Returns cropped image with just the face
    cropped_image = np.zeros((w, h, 3), dtype=np.uint8)
    cropped_image = image[y:y+h, x:x+w]

    return cropped_image
#---------------------------------------------------------

""" extracts the index of a given point """
def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index

# loading image and creating temporary images that will help during the face swap
img = cv2.imread(FILENAME)
final_image = img.copy()
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_new_face1 = np.zeros_like(img)
img_new_face2 = np.zeros_like(img)
mask = np.zeros_like(img_gray)

# Loading Face landmarks detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_FILE)



""" Face 1: Finding Landmarks """

faces = detector(img_gray)
print('num of faces', len(faces))

if len(faces) != 2:
    sys.exit('Unable to face swap')

face = faces[0]

landmarks = predictor(img_gray, face)
landmarks_points = []

for n in range(0, 68):
    x = landmarks.part(n).x
    y = landmarks.part(n).y
    landmarks_points.append((x, y))
    # cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

points = np.array(landmarks_points, np.int32)

# creating an outline of the face
convexhull = cv2.convexHull(points)
#cv2.polylines(img, [convexhull], True, (255, 0, 0), 3)
cv2.fillConvexPoly(mask, convexhull, 255)
#image containing just the face
face_image_1 = cv2.bitwise_and(img, img, mask=mask)



""" Delaunay triangulation """
rect = cv2.boundingRect(convexhull)
subdiv = cv2.Subdiv2D(rect)
subdiv.insert(landmarks_points)
triangles = subdiv.getTriangleList()
triangles = np.array(triangles, dtype=np.int32)

indexes_triangles = []
for t in triangles:
    pt1 = (t[0], t[1])
    pt2 = (t[2], t[3])
    pt3 = (t[4], t[5])
    index_pt1 = np.where((points == pt1).all(axis=1))
    index_pt1 = extract_index_nparray(index_pt1)
    index_pt2 = np.where((points == pt2).all(axis=1))
    index_pt2 = extract_index_nparray(index_pt2)
    index_pt3 = np.where((points == pt3).all(axis=1))
    index_pt3 = extract_index_nparray(index_pt3)

    if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
        triangle = [index_pt1, index_pt2, index_pt3]
        indexes_triangles.append(triangle)



""" Face 2: Finding Landmarks """

face = faces[1]

landmarks = predictor(img_gray, face)
landmarks_points2 = []
for n in range(0, 68):
    x = landmarks.part(n).x
    y = landmarks.part(n).y
    landmarks_points2.append((x, y))
    # cv2.circle(img, (x, y), 3, (0, 255, 0), -1)

points2 = np.array(landmarks_points2, np.int32)
# creating an outline of the face
convexhull2 = cv2.convexHull(points2)

# to prevent lines from showing up in the resulting image
lines_space_mask = np.zeros_like(img_gray)
lines_space_new_face = np.zeros_like(img)



""" 
Triangulation of both faces:

Taking each corresponding triangle from each face, warping it to match the other face,
then saving the results in img_new_face.
"""
for triangle_index in indexes_triangles:
    # Triangulation of the first face
    tr1_pt1 = landmarks_points[triangle_index[0]]
    tr1_pt2 = landmarks_points[triangle_index[1]]
    tr1_pt3 = landmarks_points[triangle_index[2]]
    triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)

    rect1 = cv2.boundingRect(triangle1)
    (x1, y1, w1, h1) = rect1
    cropped_triangle = img[y1: y1 + h1, x1: x1 + w1]
    cropped_tr1_mask = np.zeros((h1, w1), np.uint8)

    points = np.array([[tr1_pt1[0] - x1, tr1_pt1[1] - y1], [tr1_pt2[0] - x1, tr1_pt2[1] - y1], [tr1_pt3[0] - x1, tr1_pt3[1] - y1]], np.int32)
    cv2.fillConvexPoly(cropped_tr1_mask, points, 255)
    # cropped_triangle = cv2.bitwise_and(cropped_triangle, cropped_triangle, mask=cropped_tr1_mask)
    
    # Lines space
    cv2.line(lines_space_mask, tr1_pt1, tr1_pt2, 255)
    cv2.line(lines_space_mask, tr1_pt2, tr1_pt3, 255)
    cv2.line(lines_space_mask, tr1_pt1, tr1_pt3, 255)
    lines_space = cv2.bitwise_and(img, img, mask=lines_space_mask)

    # drawing the triangles of face1
    # cv2.line(img, tr1_pt1, tr1_pt2, (0, 0, 255), 2)
    # cv2.line(img, tr1_pt3, tr1_pt2, (0, 0, 255), 2)
    # cv2.line(img, tr1_pt1, tr1_pt3, (0, 0, 255), 2)


    # Triangulation of second face
    tr2_pt1 = landmarks_points2[triangle_index[0]]
    tr2_pt2 = landmarks_points2[triangle_index[1]]
    tr2_pt3 = landmarks_points2[triangle_index[2]]
    triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)

    rect2 = cv2.boundingRect(triangle2)
    (x2, y2, w2, h2) = rect2
    cropped_triangle2 = img[y2: y2 + h2, x2: x2 + w2]
    cropped_tr2_mask = np.zeros((h2, w2), np.uint8)

    points2 = np.array([[tr2_pt1[0] - x2, tr2_pt1[1] - y2], [tr2_pt2[0] - x2, tr2_pt2[1] - y2], [tr2_pt3[0] - x2, tr2_pt3[1] - y2]], np.int32)
    cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)

    # drawing the triangles of the face
    # cv2.line(img2, tr2_pt1, tr2_pt2, (0, 0, 255), 2)
    # cv2.line(img2, tr2_pt3, tr2_pt2, (0, 0, 255), 2)
    # cv2.line(img2, tr2_pt1, tr2_pt3, (0, 0, 255), 2)


    # Warp triangles from face1 to face2
    points = np.float32(points)
    points2 = np.float32(points2)

    M = cv2.getAffineTransform(points, points2)

    warped_triangle = cv2.warpAffine(cropped_triangle, M, (w2, h2))
    warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)

    # Reconstructing destination face
    img_new_face_rect_area = img_new_face2[y2: y2 + h2, x2: x2 + w2]
    img_new_face_rect_area_gray = cv2.cvtColor(img_new_face_rect_area, cv2.COLOR_BGR2GRAY)
    _, mask_triangles_designed = cv2.threshold(img_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
    warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)
    
    img_new_face_rect_area = cv2.add(img_new_face_rect_area, warped_triangle)
    img_new_face2[y2: y2 + h2, x2: x2 + w2] = img_new_face_rect_area


    # warp triangles from face2 to face1
    M = cv2.getAffineTransform(points2, points)

    warped_triangle = cv2.warpAffine(cropped_triangle2, M, (w1, h1))
    warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr1_mask)

    # Reconstructing destination face
    img_new_face_rect_area = img_new_face1[y1: y1 + h1, x1: x1 + w1]
    img_new_face_rect_area_gray = cv2.cvtColor(img_new_face_rect_area, cv2.COLOR_BGR2GRAY)
    _, mask_triangles_designed = cv2.threshold(img_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
    warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)
    
    img_new_face_rect_area = cv2.add(img_new_face_rect_area, warped_triangle)
    img_new_face1[y1: y1 + h1, x1: x1 + w1] = img_new_face_rect_area




# Face swapped (putting 1st face into 2nd face)
img2_face_mask = np.zeros_like(img_gray)
img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
img2_face_mask = cv2.bitwise_not(img2_head_mask)

img2_head_noface = cv2.bitwise_and(img, img, mask=img2_face_mask)

result = cv2.add(img2_head_noface, img_new_face2)

face2_rect = cv2.boundingRect(convexhull2)
face2_result_cropped = crop_image(face2_rect, result)
img2_head_mask_cropped = crop_image(face2_rect, img2_head_mask)

(x, y, w, h) = cv2.boundingRect(convexhull2)
center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))
final_image = cv2.seamlessClone(face2_result_cropped, final_image, img2_head_mask_cropped, center_face2, cv2.NORMAL_CLONE)



# Face swapped (putting 2nd face into 1st face)
img1_face_mask = np.zeros_like(img_gray)
img1_head_mask = cv2.fillConvexPoly(img1_face_mask, convexhull, 255)
img1_face_mask = cv2.bitwise_not(img1_head_mask)

img1_head_noface = cv2.bitwise_and(img, img, mask=img1_face_mask)

result = cv2.add(img1_head_noface, img_new_face1)

face1_rect = cv2.boundingRect(convexhull)
face1_result_cropped = crop_image(face1_rect, result)
img1_head_mask_cropped = crop_image(face1_rect, img1_head_mask)

(x, y, w, h) = cv2.boundingRect(convexhull)
center_face1 = (int((x + x + w) / 2), int((y + y + h) / 2))
final_image = cv2.seamlessClone(face1_result_cropped, final_image, img1_head_mask_cropped, center_face1, cv2.NORMAL_CLONE)




# Displaying and Saving the final face swap
cv2.imshow('final image', final_image)
cv2.imwrite(SAVE_IMAGE, final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

