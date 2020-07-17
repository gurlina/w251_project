# LightGAN Camera

## Running predictions on Jetson using pre-recorded video as a source  

![](earth_540_gan.gif)

The application reads frames from a pre-recorded video, runs predictions on them, and displays both input and output frames side by side in a loop.  


## Create & populate test data directories  

* Create a root folder for the project's test data (e.g., `/home/nvidia/W251/data/LightGAN`).  
* Inside the root data folder, create two subfolders named `testA` and `testB`.  
* Copy a seed image to `testB` folder (could be any image of any size).  
* Copy a video file on which you'd like to perform inference to `testA` (valid file extensions are '.avi' or '.mp4').  


**The following commands should be run from the root project directory (`w251_project`).**  

## Prepare trained model  

* Create `checkponts/enlightening` folder and copy a trained EnlightenGAN model there (e.g., `200_net_G_A.pth`).  

## Build docker image  

```
sudo docker build -t enlgan -f Dockerfile.enlgan .
```

## Start docker container in the interactive mode
```
xhost +

# make sure to map the local data directory to `/test_dataset
sudo docker run --name enlgan --net=host -e DISPLAY=$DISPLAY --rm --privileged -v /home/nvidia/W251/data/LightGAN:/test_dataset -v /tmp/.X11-unix:/tmp/.X11-unix -it enlgan
```  

## Notes  
* Currently, the script will only perform inference on the first video file it finds in `testA` directory.  

* To experiment with different input files without exiting the container, terminate the running script from inside the container (`Ctrl-C`) and overwrite the video file in `testA` folder. Then restart the script by running this command: `python scripts/script.py --predict`.  

* To save the stream of inference frames (along with their corresponding input frames) as a video file, set `--is_recording` flag in ` scripts/script.py` file to `1`, then restart the script. This will create `inference.mp4` file in the project's root directory.  

