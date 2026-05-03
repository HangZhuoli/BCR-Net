ARCH=DREB_Net_crater
EXP_ID=create_DREB_FPN_Net_detect_SSAttn_v2
DATASET=crater
INP_SHARP_OR_BLUR=SB_deblur

SHARP_DATA_DIR=../dataset/GeoMar/GeoMar-Crater-preprocess_ignore_black
BLUR_DATA_DIR=../dataset/GeoMar/GeoMar_Crater_blur/2_DeblurGAN/blur_image_ignore_black

IMG_PATH=$BLUR_DATA_DIR/Geo-val/images/004119.jpg
MODEL_PATH=./exp/detect/train/${EXP_ID}/model_90.pth
CUDA_VAL_DEVICE=0  # CPU 用 -1

python test_visual_single.py \
--exp_id $EXP_ID \
--arch $ARCH \
--dataset $DATASET \
--inp_sharp_or_blur $INP_SHARP_OR_BLUR \
--input_res 256 \
--gpus $CUDA_VAL_DEVICE \
--mode test \
--fix_res \
--load_model $MODEL_PATH \
--flip_test \
--input_img $IMG_PATH \
--sharp_data_dir $SHARP_DATA_DIR \
--blur_data_dir $BLUR_DATA_DIR
