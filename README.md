# Clothing Recommender
Search bar making using of images (dress recognition and recommendation)

## Steps to follow
1. clone the repo into a folder "x".
2. go in the directory "x", then into "dev".
3. now clone the repo - https://github.com/ultralytics/yolov5 into "dev".
4. add the "dress.yaml" file into "yolov5/data" directory.
5. replace the yolov5x.yaml with yolov5x.yaml in "yolov5/models" directory.
6. download the file "train" from the following link - "https://drive.google.com/drive/folders/1V-36pI3HeKRXwJlF_eH-rgS5RfbMNM8y?usp=sharing"
7. move the "train" folder into "dev" directory.
8. create a file "inference/images" inside the "x/dev/yolv5" folder.
9. for predection of the image put the sample image in "x/dev/yolv5/inference/images".
10. now run the "run.py" folder with the instruction mentioned in it.
11. All the recomended images are now in the saved into the "run/out" with the specific category of the clothing. [ex: "long sleeve top","shorts",....etc]