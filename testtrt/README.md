## Update EnlightenGAN files  

* Overwrite `networks.py`, `single_model.py`, `unit_network.py` in `models` folder using the file versions in this directory
* Overwrite `predict.py` script in the root directory of the repo
* Create `checkponts/enlightening` folder in the project's root directory and place the trained PyTorch model there (e.g., `200_net_G_A.pth`)

## Build docker image

```
sudo docker build -t enlgan -f Dockerfile.enlgan .
```

## Start docker container in the interactive mode
```
cd <root project directory>
sudo docker run --privileged --rm -v "$PWD":/EnlightenGAN --memory-swap --memory-swappiness 100 -ti enlgan bash
```  

## Copy test images and modified `torch2trt` files ([cat.py](https://github.com/gurlina/w251_project/blob/master/testtrt/cat.py)) to the docker container (using a different terminal window)

```
# get the container id
sudo docker ps -a

sudo docker cp image_1.png <container_id>:/test_dataset/testA/
sudo docker cp image_2.png <container_id>:/test_dataset/testB/
sudo docker cp cat.py  <container_id>:/torch2trt/torch2trt/converters/
```  

## Inside the container

```
# register the modified version of the concatination converter
cd /torch2trt
python setup.py install

# run the prediction on the test image
cd /EnlightenGAN
python scripts/script.py --predict
```

## Notes 

* The docker container uses [torch2trt ](https://github.com/NVIDIA-AI-IOT/torch2trt) PyTorch to TensorRT converter.

* There was a bug in their `cat.py` converter that prevented the TRT engine from correctly concatenating inputs. This bug is now fixed in the checked in version of the file here.

* Some of the PyTorch operations employed by EnlightenGAN network are not supported by `torch2trt`. In this case, the conversion engine logs a warning and is supposed to use the default PyTorch implementation. There are currently 3 unsupported functions:

  ```
  Warning: Encountered known unsupported method torch.Tensor.__hash__
  Warning: Encountered known unsupported method torch.Tensor.get_device
  Warning: Encountered known unsupported method torch.nn.functional.interpolate
  ```  

* `torch2trt` provides an experimental plugin for `interpolate` operation but any attempts to use it resulted in errors.  

* Similarly, setting `use_onnx` flag to `True` while calling the converter resulted in failure (see `torch2trt(...)` function call in [single_model.py](https://github.com/gurlina/w251_project/blob/master/testtrt/single_model.py))

* The prediction script curently terminates in errors:  
  ```
  [TensorRT] ERROR: Unused Input: input_0
  [TensorRT] ERROR: Unused Input: input_1
  ```



