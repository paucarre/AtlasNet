from __future__ import print_function
import argparse
import random
import numpy as np
import torch
import sys
sys.path.append('./auxiliary/')
from dataset import *
from model import *
from utils import *
from ply import *
import os
import json
import pandas as pd

# =============PARAMETERS======================================== #
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='input PLY file path')
parser.add_argument('--output', type=str, default="output folder path")
parser.add_argument('--model', type=str, default = 'trained_models/ae_atlasnet_25.pth',  help='yuor path to the trained model')
parser.add_argument('--num_points', type=int, default = 2500,  help='number of points fed to poitnet')
parser.add_argument('--gen_points', type=int, default = 25000,  help='number of points to generate, put 30000 for high quality mesh, 2500 for quantitative comparison with the baseline')
parser.add_argument('--nb_primitives', type=int, default = 25,  help='number of primitives')

opt = parser.parse_args()
print(f'Arguments: {opt}')

# ===================CREATE network================================= #
network = AE_AtlasNet(num_points = opt.num_points, nb_primitives = opt.nb_primitives)
network.apply(weights_init)
network.load_state_dict(torch.load(opt.model, map_location='cpu' ))
network.eval()
# ========================================================== #


# =============DEFINE ATLAS GRID ======================================== #
grain = int(np.sqrt(opt.gen_points/opt.nb_primitives))-1
grain = grain*1.0
print(f"Gen Points: {opt.gen_points}\nGrain: {grain}\nPrimitives: {opt.nb_primitives}")

#generate regular grid
faces = []
vertices = []
face_colors = []
vertex_colors = []
colors = get_colors(opt.nb_primitives)

for i in range(0,int(grain + 1 )):
        for j in range(0,int(grain + 1 )):
            vertices.append([i/grain,j/grain])

for prim in range(0,opt.nb_primitives):
    for i in range(0,int(grain + 1)):
        for j in range(0,int(grain + 1)):
            vertex_colors.append(colors[prim])

    for i in range(1,int(grain + 1)):
        for j in range(0,(int(grain + 1)-1)):
            faces.append([(grain+1)*(grain+1)*prim + j+(grain+1)*i, (grain+1)*(grain+1)*prim + j+(grain+1)*i + 1, (grain+1)*(grain+1)*prim + j+(grain+1)*(i-1)])
    for i in range(0,(int((grain+1))-1)):
        for j in range(1,int((grain+1))):
            faces.append([(grain+1)*(grain+1)*prim + j+(grain+1)*i, (grain+1)*(grain+1)*prim + j+(grain+1)*i - 1, (grain+1)*(grain+1)*prim + j+(grain+1)*(i+1)])
grid = [vertices for i in range(0,opt.nb_primitives)]
grid_pytorch = torch.Tensor(int(opt.nb_primitives*(grain+1)*(grain+1)),2)
for i in range(opt.nb_primitives):
    for j in range(int((grain+1)*(grain+1))):
        grid_pytorch[int(j + (grain+1)*(grain+1)*i),0] = vertices[j][0]
        grid_pytorch[int(j + (grain+1)*(grain+1)*i),1] = vertices[j][1]

print("grain", grain, 'number vertices', len(vertices)*opt.nb_primitives)

# ========================================================== #

# =============TESTING LOOP======================================== #
#Iterate on the data
with torch.no_grad():

    #_, points, cat , objpath, fn = data
    points = ShapeNet.load_point_set(opt.input, npoints=opt.num_points).unsqueeze(0)

    points = points.transpose(2,1).contiguous()
    pointsReconstructed  = network.forward_inference(points, grid)

    if not os.path.exists(str(opt.output) ):
        os.mkdir(str(opt.output) )
        print('created dir', str(opt.output) )

    write_ply(filename = opt.output + "/" + os.path.basename(opt.input)+"_GT", points=pd.DataFrame(points.transpose(2,1).contiguous().cpu().data.squeeze().numpy()), as_text=True)
    b = np.zeros((len(faces),4)) + 3
    b[:,1:] = np.array(faces)
    write_ply(filename = opt.output + "/" + os.path.basename(opt.input)+"_gen" + "_grain_" + str(int(opt.gen_points)), points=pd.DataFrame(torch.cat((pointsReconstructed.cpu().data.squeeze(), grid_pytorch), 1).numpy()), as_text=True, text=True, faces = pd.DataFrame(b.astype(int)))
