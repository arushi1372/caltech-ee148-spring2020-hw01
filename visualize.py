import json
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

def visualize(data):
    data_path = 'RedLights2011_Medium'
    save_path = 'visualizations/'
    for key, value in data.items():
        im = Image.open(os.path.join(data_path, key))
        # Display the image
        plt.imshow(im)
        # Get the current reference
        ax = plt.gca()
        # Create a Rectangle patch
        for box in value:
            tlx, tly, brx, bry = box[0], box[1], box[2], box[3]
            rect = Rectangle((tlx, bry), (brx - tlx), (tly - bry), 
                              linewidth=1, edgecolor='r', facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(rect)
        plt.savefig(save_path + key)
        plt.close()

with open('hw01_preds/preds.json') as f:
  data = json.load(f)
visualize(data)