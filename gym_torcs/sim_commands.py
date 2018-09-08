import os
import re
import numpy as np
PATH_TO_CONFIG_DIR = "/usr/local/share/games/torcs/config"
TRACK_SELECT_FILE = os.path.join(PATH_TO_CONFIG_DIR, "raceman/practice.xml")
SCREEN_FILE = os.path.join(PATH_TO_CONFIG_DIR, "screen.xml")
TRACK_DIR = "/usr/local/share/games/torcs/tracks"
PATH_TO_CONFIG_DIR = "/usr/local/share/games/torcs/config"
TRACK_SELECT_FILE = os.path.join(PATH_TO_CONFIG_DIR, "raceman/practice.xml")
SCREEN_FILE = os.path.join(PATH_TO_CONFIG_DIR, "screen.xml")
DB_DIRS = os.path.join(os.path.dirname(os.path.abspath("__FILE__")), "databases")

def set_track(t):
    with open(TRACK_SELECT_FILE, 'r') as f:
        text = f.read()
    text = re.sub(r"""<attstr name="name" val=".+"/>
      <attstr name="category" val=".+"/>""", r"""<attstr name="name" val="%s"/>
      <attstr name="category" val="road"/>"""%t, text)

    with open(TRACK_SELECT_FILE, 'w') as f:
        f.write(text)

def set_sim_size(x,y):
    with open(SCREEN_FILE, 'r') as f:
        text = f.read()
    text = re.sub(r'<attnum name="x" val="\d+"\/>', '<attnum name="x" val="%s"/>'%x, text)
    text = re.sub(r'<attnum name="y" val="\d+"\/>', '<attnum name="y" val="%s"/>'%y, text)
    text = re.sub(r'<attnum name="window width" val="\d+"\/>','<attnum name="window width" val="%s"/>'%x, text)
    text = re.sub(r'<attnum name="window height" val="\d+"\/>','<attnum name="window height" val="%s"/>'%y, text)
    with open(SCREEN_FILE, 'w') as f:
        f.write(text)

def obs_vision_to_image_rgb(obs_image_vec):
    image_vec =  obs_image_vec
    w = 64
    h = 64
    nc = 3
    image_vec = np.flipud(np.array(image_vec).astype(np.uint8).reshape([w, h, 3]))
    return image_vec


# def simulator_setup_teardown(func, *args, **kwargs):
#     def wrapped_func(args, kwargs):
#         try:
#             func(args, kwargs)
#         except Exception as e:
#             print(e)
#         finally:
#             restore_sim_defaults()
