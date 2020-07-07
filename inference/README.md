Inference scripts  

Note: when running thes .py scripts in dockere, please move these scripts to the EnlightenGAN directory (one level up from this inference sub directory)  


Steps to reproduce / deploy on Jetson: 
1. create docker image `docker build -t infer_from_train -f Dockerfile.infer_from_train .`
2. type `xhost +` on the terminal locally
3. clone enlightenGAN repo locally 
4. copy test_script.py and video_predict.py into /EnlightenGAN/ on your docker
5. copy seed_file.png and test.avi into /EnlightenGAN/
5. run docker container `sudo docker run --privileged --rm -e DISPLAY=$DISPLAY -v "/tmp/.x11-unix/:/tmp/.x11-unix" --net=host --ipc=host -ti --entrypoint sh infer_from_train`
6. once the docker container is up, cd into the EnlightenGAN directory and run `python test_script.py` to invoke video_predict.py. This script will read in the test.avi video (image from my pantry) frame by frame and lighten them. You should see the original dark input and the brightened output appear in cv2.imshow windows. Note they tend to stack on top of each other, so move the output window to see the input one underneath.  


Areas for optimization:
1. Currently the data_loader is created each time a new frame comes in from a video. It looks like this is the most time consuming part
2. For some reason, the data loader things there are 46 images for each frame (should only be 1). I worked around it for now by breaking out of the for loop past the first iteration through, but again this could explain why the data loader is taking more time than expected
