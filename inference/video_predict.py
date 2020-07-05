import cv2 as cv
import numpy as np
import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html
from util import util

def brighten_frame(img, frame_no, opt):
    print("Frame Number:", frame_no)
    input_path = '/notebooks/test_dataset/testA/input' + str(frame_no) + '.png'
    cv.imwrite(input_path,img)
    print("Writing image to test dataset location testA")

    print("Creating Data Loader")
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()

    # For some reason I am getting 46 images in the dataset each time I load in one image

    print("Number of images in dataset:", len(dataset))
    for i, data in enumerate(dataset):
        if i > 0:
            break
        model.set_input(data)
        visuals = model.predict()
        img_path = model.get_image_paths()
        print('process image... %s' % img_path)

        # NK added
        image_dir = webpage.get_image_dir()
        short_path = os.path.basename(img_path[0])
        name = os.path.splitext(short_path)[0]

        for label, image_numpy in visuals.items():
            image_name = '%s.jpg' % ('frame' + str(frame_no))
            save_path = os.path.join("../test_dataset/testB/", image_name)
            util.save_image(image_numpy, save_path)

    print("all done! cleaning up file", input_path)
    os.remove(input_path)
    return save_path



cap = cv.VideoCapture('test.avi')
i = 0

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

model = create_model(opt)

web_dir = os.path.join("./ablation/", opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

while cap.isOpened():
    ret, frame = cap.read()

    if cv.waitKey(1) & 0xFF == ord('q'):
        break


    # print(frame.shape)
    cv.imshow("input", frame)

    output_path = brighten_frame(frame, i, opt)

    print("Output Path: ", output_path)
    output_image = cv.imread(output_path)
    cv.imshow("output", output_image)

    i += 1
    print(i, "frame")

print(i, "frames")
cap.release()
cv.destroyAllWindows()

