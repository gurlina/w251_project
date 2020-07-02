import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
#from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html
from util import util

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
#visualizer = Visualizer(opt)
# create website
web_dir = os.path.join("./ablation/", opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test
print(len(dataset))
for i, data in enumerate(dataset):
    model.set_input(data)
    visuals = model.predict()
    img_path = model.get_image_paths()
    print('process image... %s' % img_path)
    
    # NK added
    image_dir = webpage.get_image_dir()
    short_path = os.path.basename(img_path[0])
    name = os.path.splitext(short_path)[0]

    for label, image_numpy in visuals.items():
            image_name = '%s.jpg' % (name)
            save_path = os.path.join("../test_dataset/testB/", "updated_" + image_name)
            util.save_image(image_numpy, save_path)
#    visualizer.save_images(webpage, visuals, img_path)

#webpage.save()
