import os
import dnnlib.tflib as tflib
import math
import moviepy.editor
from numpy import linalg
import numpy as np
import pickle
import PIL.Image

tflib.init_tf()
_G, _D, Gs = pickle.load(open("gan_network/pleo4.pkl", "rb"))

rnd = np.random
latents_a = rnd.randn(1, Gs.input_shape[1])
latents_b = rnd.randn(1, Gs.input_shape[1])
latents_c = rnd.randn(1, Gs.input_shape[1])

def circ_generator(latents_interpolate):
    radius = 40.0

    latents_axis_x = (latents_a - latents_b).flatten() / linalg.norm(latents_a - latents_b)
    latents_axis_y = (latents_a - latents_c).flatten() / linalg.norm(latents_a - latents_c)

    latents_x = math.sin(math.pi * 2.0 * latents_interpolate) * radius
    latents_y = math.cos(math.pi * 2.0 * latents_interpolate) * radius

    latents = latents_a + latents_x * latents_axis_x + latents_y * latents_axis_y
    return latents

def mse(x, y):
    return (np.square(x - y)).mean()

def generate_from_generator_adaptive(gen_func, class_label):
    max_step = 1.0
    current_pos = 0.0

    change_min = 10.0
    change_max = 11.0

    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)

    current_latent = gen_func(current_pos)   
    labels = np.zeros([current_latent.shape[0]] + Gs.input_shapes[1][1:])
    labels[:,class_label]=1
    current_image = Gs.run(current_latent, labels, truncation_psi=0.5, randomize_noise=False, output_transform=fmt)[0]
    array_list = []

    video_length = 1.0
    while(current_pos < video_length):
        array_list.append(current_image)

        lower = current_pos
        upper = current_pos + max_step
        current_pos = (upper + lower) / 2.0

        current_latent = gen_func(current_pos)
        #print(current_pos)
        current_image = images = Gs.run(current_latent, labels, truncation_psi=0.5, randomize_noise=False, output_transform=fmt)[0]
        current_mse = mse(array_list[-1], current_image)

        while current_mse < change_min or current_mse > change_max:
            if current_mse < change_min:
                lower = current_pos
                current_pos = (upper + lower) / 2.0

            if current_mse > change_max:
                upper = current_pos
                current_pos = (upper + lower) / 2.0


            current_latent = gen_func(current_pos)
            current_image = images = Gs.run(current_latent, labels, truncation_psi=0.5, randomize_noise=False, output_transform=fmt)[0]
            current_mse = mse(array_list[-1], current_image)
        print(current_pos, current_mse)
    return array_list

label = []
for fold in sorted(os.listdir('datasets/pleo224')):
  label.append(fold)

for idx in range(0,7):
  class_label = idx
  frames = generate_from_generator_adaptive(circ_generator,idx)
  i=0
  for frame in frames:
    image = PIL.Image.fromarray(frame, 'RGB')
    display(image)
    #os.makedirs('fakes', exist_ok=True)
    os.makedirs('fakes4_psi05/'+label[idx], exist_ok=True)
    png_filename = os.path.join('fakes4_psi05',label[idx], 'example-'+str(i)+'.png')
    image.save(png_filename)
    i+=1
