[Project Page](https://yoterel.github.io/PractiLight-project-page/) | [Paper]() | [Supplementary]()

# PractiLight
PractiLight offers a light-weight solution for relighting a prompt-generated image using a simple control signal - direct illumination.
On contrary to prior work, we support relighting various image domains, including portraits, sketchs, landscapes, anime, and many more !

## Dependencies
 - intrinsic (https://github.com/compphoto/Intrinsic)
 - diffusers
 - transformers
 - torch
 - yaml
 - opencv
 - gsoup
 - bpy (optional, used for programatically creating a control signal)

Note: you need these packages installed, we recommend a virtual environemnt such as conda or mamba to do so.

## Usage

### Step 1

Edit the prompt and all other settings in [config.yml](https://github.com/yoterel/practi_light/blob/main/config.yml), and run:

`python relight.py config.yml`

This will create an output file (it is a 2D disparity map) that you can load on blender using our addon.

### Step 2

Open [booth.blend](https://github.com/yoterel/practi_light/blob/main/booth.blend) with Blender 3.6, and follow the animation bellow:

![example](https://github.com/yoterel/practi_light/blob/main/resource/example.gif)

Note: pressing the "run" button in the begining simply loads the addon, then the "Load Depth Map" will appear in the menu.

You can swap the light type, and change the roughness in the node editor. In fact, this way of creating a control signal is just a suggestion. Feel free to use any input as control. For better results it should resemble a direct irradiance map (this is Blender's output in the above video).

### Step 3
After you render, save the file to disk, and this is your new control signal. Now edit the parameter called "control_signal" in [config.yml](https://github.com/yoterel/practi_light/blob/main/config.yml), and make it point to the rendered file from blender from step 2.

### Step 4
Finally, run the same command again:

`python relight.py config.yml`

Which will generate the relit result.

## Todos
- [x] Manual illumination
- [ ] Support sequence of control signals (from folder)
- [ ] Automatic illumination
- [ ] Reproduction of dataset / benchmark
