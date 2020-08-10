# Face Swap using Python

There are two different versions of this faceswap project:
1. Image version, which just takes an image stored in your computer (you will have to edit the source code itself) and detects faces and does a face swap if only 2 faces are found.
2. Video version, accesses your computer's video feed to display a resulting face swap if two faces are found.

## Resources used
### Video
* Imports cv2 and numpy.
* Haarcascades
    * haarcascade_frontalface_default.xml
    * haarcascade_eye.xml

### Image
* Imports cv2, sys, numpy, and dlib
* Adapted code from pysource's face swap method using Delaunay triangulation https://pysource.com/2019/05/28/face-swapping-explained-in-8-steps-opencv-with-python/
* Predictor File
    ** shape_predictor_68_face_landmarks.dat