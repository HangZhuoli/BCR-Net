ARCH=DREB_Net
EXP_ID=create_DREB_FPN_Net_detect_SSAttn_v2
DATASET=crater
INP_SHARP_OR_BLUR=SB_deblur
SHARP_DATA_DIR=../dataset/GeoMar/GeoMar-Crater-preprocess_ignore_black  #注意数据集的存放的对应的路径
BLUR_DATA_DIR=../dataset/GeoMar/GeoMar_Crater_blur/2_DeblurGAN/blur_image_ignore_black
BEST_MODEL=./exp/detect/train/${EXP_ID}/model_best.pth
LAST_MODEL=./exp/detect/train/${EXP_ID}/model_last.pth
CUDA_TRAIN_DEVICE=0

CUDA_VISIBLE_DEVICES=$CUDA_TRAIN_DEVICE python main.py \
--exp_id $EXP_ID \
--arch $ARCH \
--dataset $DATASET \
--inp_sharp_or_blur $INP_SHARP_OR_BLUR \
--sharp_data_dir $SHARP_DATA_DIR \
--blur_data_dir $BLUR_DATA_DIR \
--input_res 1024 \
--mode train \
--batch_size 16 \
--master_batch 4 \
--lr 1e-3  \
--num_epochs 10 \
--gpus $CUDA_TRAIN_DEVICE