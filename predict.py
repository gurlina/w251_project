import cv2
import numpy as np

from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

# optional: record predicted frames as video
recording = opt.is_recording
fps = 20
output_file = "inference.mp4"
is_recorded = False
videoWriter = None

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)

# process input video in endless loop
while(True):

    for i, data in enumerate(dataset):
        model.set_input(data)
        visuals = model.predict()
         
        # the iterator alternates between real and fake images
        for label, image_numpy in visuals.items():
            # save a copy of the real frame
            if label.startswith('real'):
                record_frame = image_numpy.copy()
            else:
                # concatenate the inferred frame with the real one (horizontally)
                record_frame = np.concatenate((record_frame, image_numpy), axis=1)

                # OpenCV expects BGR color ordering, so we need to perform RGB2BGR conversion
                record_frame = cv2.cvtColor(record_frame, cv2.COLOR_RGB2BGR)
                cv2.imshow("LightGAN Camera (Input | Output)", record_frame)
                cv2.waitKey(1)
                
                # if the recording flag is set, generate video file
                if recording and not is_recorded:
                    if videoWriter is None:
                        print("Generating video recording...")
                        size = (record_frame.shape[1], record_frame.shape[0])
                        videoWriter = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc('X','2','6','4'), fps, size)
                    else:
                        videoWriter.write(record_frame) 
    
    # save recording only once
    if recording and not is_recorded: 
        print("Recording saved in", output_file)       
        is_recorded = True

    # check if any key was pressed; quit the loop on 'q'	
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break;

cv2.destroyAllWindows()
