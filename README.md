# Face Swap using Python

There are three different versions of this faceswap project:
1. Image version, which just takes an image stored in your computer (you will have to edit the source code itself) and detects faces and does a face swap if only 2 faces are found.
2. Video version, accesses your computer's video feed to display a resulting face swap if two faces are found.
3. Cat version, where a photo of a face and a photo of a cat face must be given. The human face will be swapped with the cat face.

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

### Cat Image
* Imports cv2 and numpy
* Haarcascades
    * haarcascade_frontalface_default.xml
    * haarcascade_frontalcatface_extended.xml

**Note:** The haarcascade files can be found with cv2 installation, but they are included in the haarcascade_files folder for convenience.

**Note:** Press 'q' to quit programs