#!/bin/sh

# python neural_style.py -gpu c -print_iter 1
# python neural_style.py -gpu 0 -print_iter 1
# python neural_style.py -gpu 0 -backend cudnn -print_iter 1

# python neural_style.py \
#   -style_image /home/acworks/Documents/playground/ostagram/style-1.jpg \
#   -content_image /home/acworks/Documents/playground/ostagram/content-1.jpg

# python neural_style.py \
#   -style_image /home/acworks/Documents/playground/ostagram/style-1.jpg \
#   -content_image /home/acworks/Documents/playground/ostagram/content-1.jpg \
#   -output_image out1.png \
#   -model_file models/nin_imagenet.pth \
#   -gpu 1 \
#   -backend cudnn \
#   -cudnn_autotune \
#   -num_iterations 1000 \
#   -content_layers relu0,relu3,relu7,relu12 \
#   -style_layers relu0,relu3,relu7,relu12 \
#   -content_weight 10 \
#   -style_weight 500 \
#   -image_size 512 \
#   -optimizer adam

CUDA_VISIBLE_DEVICES=0 python neural_style.py \
  -style_image /home/acworks/Documents/playground/ostagram/style-1.jpg \
  -content_image /home/acworks/Documents/playground/ostagram/content-1.jpg \
  -init image \
  -output_image out1.png \
  -gpu 0 \
  -backend cudnn \
  -cudnn_autotune \
  -content_weight 0.5 \
  -style_weight 500 \
  -image_size 512 \
  -seed 123
