#!/bin/sh

CONTENT_IMAGE=/home/acworks/Documents/playground/ostagram/content-24.jpg
STYLE_IMAGE=/home/acworks/Documents/playground/ostagram/style-4.jpg
STYLE_WEIGHT=3000
GPU=1

# Step 1
python neural_style.py -tv_weight 0 -output_image mitom1.png -init image \
-image_size 512 -num_iterations 1000 -content_weight 0.5 -style_weight $STYLE_WEIGHT \
-save_iter 0 -backend cudnn -cudnn_autotune \
-content_image $CONTENT_IMAGE -style_image $STYLE_IMAGE \
-gpu $GPU

# Step 2
python neural_style.py -tv_weight 0 -output_image mitom2.png -init image \
-init_image mitom1.png -image_size 720 -num_iterations 1000 -save_iter 0 \
-content_weight 0.5 -style_weight $STYLE_WEIGHT -backend cudnn -cudnn_autotune \
-content_image $CONTENT_IMAGE -style_image $STYLE_IMAGE \
-gpu $GPU

# Step 3
python neural_style.py -tv_weight 0 -output_image mitom3.png -init image \
-init_image mitom2.png -image_size 1024 -num_iterations 500 -save_iter 0 \
-content_weight 0.5 -style_weight $STYLE_WEIGHT -backend cudnn -cudnn_autotune \
-content_image $CONTENT_IMAGE -style_image $STYLE_IMAGE \
-gpu $GPU

# Step 4
python neural_style.py -tv_weight 0 -output_image mitom4.png -init image \
-init_image mitom3.png -image_size 1536 -num_iterations 200 -save_iter 0 \
-content_weight 0.5 -style_weight $STYLE_WEIGHT -backend cudnn -cudnn_autotune \
-content_image $CONTENT_IMAGE -style_image $STYLE_IMAGE \
-optimizer adam \
-gpu $GPU


# Step 5
python neural_style.py -tv_weight 0 -output_image mitom5.png -init image \
-init_image mitom4.png -image_size 2048 -num_iterations 100 -save_iter 0 \
-content_weight 0.5 -style_weight $STYLE_WEIGHT -backend cudnn -cudnn_autotune -optimizer adam \
-content_image $CONTENT_IMAGE -style_image $STYLE_IMAGE \
-gpu $GPU

# Step 6
python neural_style.py -tv_weight 0 -output_image mitom6.png -init image \
-init_image mitom5.png -image_size 2536 -num_iterations 50 -save_iter 0 \
-content_weight 0.5 -style_weight $STYLE_WEIGHT -backend cudnn -cudnn_autotune -optimizer adam \
-content_image $CONTENT_IMAGE -style_image $STYLE_IMAGE \
-model_file models/nin_imagenet.pth \
-content_layers relu0,relu3,relu7,relu12 \
-style_layers relu0,relu3,relu7,relu12 \
-gpu $GPU

# Step 7
python neural_style.py -tv_weight 0 -output_image mitom7.png -init image \
-init_image mitom6.png -image_size 3000 -num_iterations 50 -save_iter 0 \
-content_weight 0.5 -style_weight $STYLE_WEIGHT -backend cudnn -cudnn_autotune -optimizer adam \
-content_image $CONTENT_IMAGE -style_image $STYLE_IMAGE \
-model_file models/nin_imagenet.pth \
-content_layers relu0,relu3,relu7,relu12 \
-style_layers relu0,relu3,relu7,relu12 \
-gpu $GPU
