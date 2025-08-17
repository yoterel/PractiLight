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
 - bpy
 - gsoup

## Usage

Edit the prompt and all other settings in [config.yml](https://github.com/yoterel/practi_light/blob/main/config.yml), and run:

`python relight.py config.yml`

Here is an example animation showing how to create a control signal in Blender 3.6 (open [booth.blend](https://github.com/yoterel/practi_light/blob/main/booth.blend)):

![example](https://github.com/yoterel/practi_light/blob/main/resource/example.gif)

You can swap the light type, and change the roughness in the node editor. In fact, this way of creating a control signal is just a suggestion. Feel free to use any input as control. For better results it should resemble a direct irradiance map (this is Blender's output in the above example).

## 

## Todos
- [x] Manual illumination
- [ ] Support sequence of control signals (from folder)
- [ ] Automatic illumination
- [ ] Reproduction of dataset / benchmark
