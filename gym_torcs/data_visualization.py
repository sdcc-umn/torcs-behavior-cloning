import argparse as AP
import pandas as pd
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image

def load_data(config):
    data_df = pd.read_csv(os.path.join(config.data_dir, 'db.csv'))  # TODO: do rename this to 'driving log' or something else more informative than 'db'
    # data_df.hist(column='ctrl', bins=100)
    # plt.show()

    # data_df = data_df[data_df.ctrl>5]
    # data_df.hist(column='ctrl', bins=100)
    # plt.show()

    data_df = data_df[abs(data_df.ctrl)>0.0001]
    data_df.hist(column='ctrl', bins=100)
    plt.show()


    img_paths=data_df.image.tolist()
    for i in range(len(data_df)):
        sample = img_paths[i]
        img = load_image(sample)
        plt.imshow(img)
        plt.show()



def load_image(image_path):
    return mpimg.imread(image_path)


if __name__ == "__main__":
    parser = AP.ArgumentParser()
    parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='.')
    args = parser.parse_args()
    load_data(args)
