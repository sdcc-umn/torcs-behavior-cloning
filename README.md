# simulator-trials
Experiments conducted with models run on simulators

# How to use
## Python dependencies
- Install anaconda python3 (https://www.anaconda.com/download/#linux)
- create a virtual environment with Conda, Anaconda's CLI tool:
`conda env create -f environment.yml -n sdcc`. If you get an error at the end, don't worry about it. (TODO: FIX)
- activate the environment `source activate sdcc`
- install remaining packages through pip
`pip install -r requirements.txt`

## Dependencies for the Simulator
- x-automation7: `sudo apt install xautomation`
- other dependencies: `sudo apt install libglib2.0-dev  libgl1-mesa-dev libglu1-mesa-dev  freeglut3-dev  libplib-dev  libopenal-dev libalut-dev libxi-dev libxmu-dev libxrender-dev  libxrandr-dev libpng12-dev ffmpeg`

## Configuring and installing the simulator
- cd into gym-torcs/vtorcs-RL-color
- `./configure`
- `make`
- `sudo make install`
- `sudo make datainstall`
- Go back up a directory (to gym-torcs).
- Configure permissions with `sudo ./configure_permissions`

## Recording Images with the simulator
- cd to gym-torcs directory
-`python snakeoil3_gym.py`
- That will record images in to `./databases/<track>/<trial>/imgs` directory.
- To make images into a movie, use fmpeg on the directory: `Ffmpeg -i <path>/imgs/%05d.png video.webm`
  
## Training the agent
- be in gym-torcs directory. 
- assumes you have already recorded some data somewhere.
- `python main.py -d ./databases/<track>/<trial>`
  
