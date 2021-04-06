import os
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def read_prototype_10():
    ''' 
    Read in an example of a red light, selected from image 10.
    '''
    data_path = 'RedLights2011_Medium'
    I = Image.open(os.path.join(data_path,"RL-010.jpg"))
    I = np.asarray(I)
    proto_red = I[25:91,320:349]
    # proto_red = I[:,:,0][25:91,320:349] # only using red channel
    return proto_red

def smooth(y, box_pts):
    '''
    Smooth a given set of datapoints.
    '''
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def normalize(a):
    return a.flatten()/np.linalg.norm(a.flatten())

def detect_red_light(I):
    '''
    This function takes a numpy array <I> and returns a list <bounding_boxes>.
    The list <bounding_boxes> should have one element for each red light in the 
    image. Each element of <bounding_boxes> should itself be a list, containing 
    four integers that specify a bounding box: the row and column index of the 
    top left corner and the row and column index of the bottom right corner (in
    that order). See the code below for an example.
    
    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''

    bounding_boxes = [] # This should be a list of lists, each of length 4. See format example below. 
    
    example = read_prototype_10()

    windows = np.lib.stride_tricks.sliding_window_view(I, example.shape)
    # windows = np.lib.stride_tricks.sliding_window_view(I[:,:,0], example.shape) # only using red channel
    lst = []
    for col_ind, axis1 in enumerate(windows):
        for row_ind, window in enumerate(axis1):
            lst.append((np.inner(normalize(window).flatten(), normalize(example).flatten()), 
                [row_ind, col_ind, row_ind + example.shape[1], col_ind + example.shape[0]]))
    
    convs = [x[0] for x in lst] # get all convolutions
    smoothed = smooth(convs, 8) # smooth convolutions
    peaks = find_peaks(smoothed, height=0.88)[0] # find peaks above threshold 0.88
    for idx in peaks:
        bounding_boxes.append(lst[idx][1])
    
    for i in range(len(bounding_boxes)):
        assert len(bounding_boxes[i]) == 4
    
    return bounding_boxes

# set the path to the downloaded data: 
data_path = 'RedLights2011_Medium'

# set a path for saving predictions: 
preds_path = 'hw01_preds' 
os.makedirs(preds_path,exist_ok=True) # create directory if needed 

# get sorted list of files: 
file_names = sorted(os.listdir(data_path)) 

# remove any non-JPEG files: 
file_names = [f for f in file_names if '.jpg' in f] 

preds = {}

for i in range(len(file_names)):
    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names[i]))
    # convert to numpy array:
    I = np.asarray(I)

    preds[file_names[i]] = detect_red_light(I)

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds.json'),'w') as f:
    json.dump(preds,f)