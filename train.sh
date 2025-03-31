# recommended paddle.__version__ == 2.0.0
####################################### ----------------------------- 检测 --------------------------------- #################################
# 训练
python3 -m paddle.distributed.launch --log_dir=./debug/ --gpus '0,1,2,3,4,5,6,7'  tools/train.py -c configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_teacher.yml

# 推理
## 1. 采用 训练模型直接推理
python tools/infer_det.py -c configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_teacher.yml -o Global.checkpoints=./output/ch_PP-OCRv4-teacher-9-9/iter_epoch_60.pdparams PostProcess.unclip_ratio=0.8 PostProcess.box_thresh=0.4  PostProcess.box_type=poly   Global.infer_img=./train_data/det_bookspine_text/val
python tools/infer_det.py -c configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_teacher.yml -o Global.checkpoints=./ch_PP-OCRv4_det_server_train/best_accuracy.pdparams PostProcess.unclip_ratio=0.8 PostProcess.box_thresh=0.4  PostProcess.box_type=poly   Global.infer_img=./train_data/det_bookspine_text/val

## 2. 采用 测试模型推理
### a. 模型转换
python tools/export_model.py -c configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_teacher.yml -o Global.checkpoints=./output/ch_PP-OCRv4-teacher-9-9/iter_epoch_60.pdparams Global.save_inference_dir=./inference/iter_epoch_60-9-9/
### b. 推理
python tools/infer/predict_det.py --image_dir=train_data/det_bookspine_text/val --det_model_dir=./inference/iter_epoch_60-9-9/  --det_db_unclip_ratio=0.4 --det_db_box_thresh=0.4




### 3. 模型评估
python tools/eval.py  -c configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_teacher.yml -o Global.checkpoints="./ch_PP-OCRv4_det_server_train/best_accuracy"
python tools/eval.py  -c configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_teacher.yml -o Global.checkpoints="./output/ch_PP-OCRv4-teacher-9-9/iter_epoch_500"


# 模型转换为 onnx
paddle2onnx --model_dir ./inference/iter_epoch_60-9-9 \
--model_filename inference.pdmodel \
--params_filename inference.pdiparams \
--save_file ./inference/det_spine_text_loc_onnx_iter_epoch_60-9-9/model.onnx \
--opset_version 10 \
--enable_onnx_checker True
####################################### ----------------------------- 识别 --------------------------------- #################################
# 训练
python tools/train.py -c configs/rec/PP-OCRv4/ch_PP-OCRv4_rec.yml
python tools/train.py -c configs/rec/PP-OCRv4/ch_PP-OCRv4_rec_hgnet.yml
nohup python -m paddle.distributed.launch --log_dir=./debug/ --gpus '0,1,2,3' tools/train.py -c configs/rec/PP-OCRv4/ch_PP-OCRv4_rec_hgnet.yml >train-10-16-2.log &

# 测试
## accuracy:0.7013574660633484  字符级别的准确率：0.9197 
 python tools/infer_rec_cal_accuracy.py -c configs/rec/PP-OCRv4/ch_PP-OCRv4_rec_hgnet.yml -o Global.checkpoints=./ch_PP-OCRv4_rec_server_train/best_accuracy.pdparams  Global.use_gpu=True  Global.infer_img=train_data/rec/test 
## accuracy:0.8027149321266969  字符级别的准确率：0.9505
 python tools/infer_rec_cal_accuracy.py -c configs/rec/PP-OCRv4/ch_PP-OCRv4_rec_hgnet.yml -o Global.checkpoints=./output/rec_ppocr_v4_hgnet2/best_accuracy.pdparams  Global.use_gpu=True  Global.infer_img=train_data/rec/test


### a. 模型转换
python tools/export_model.py -c configs/rec/PP-OCRv4/ch_PP-OCRv4_rec_hgnet.yml -o Global.checkpoints=./output/rec_ppocr_v4_hgnet2/best_accuracy.pdparams Global.save_inference_dir=./inference/ch_PP-OCRv4_rec_hgnet_2024_10_17/






# =============================训练并查看效果=========================================== 
# 训练
nohup python -u tools/train.py -c configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_teacher.yml >train-12-6.log &
# 推理（使用训练模型）
python -u tools/infer_det.py -c configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_teacher.yml

###############################模型部署######################################
# step1 训练模型 ---> 推理模型
python tools/export_model.py -c configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_teacher.yml -o Global.checkpoints=./output/ch_PP-OCRv4-teacher-2025-3-31/best_accuracy.pdparams   Global.save_inference_dir=./inference/ch_PP-OCRv4-teacher-2025-3-31/
# step2 推理模型 ---> onnx
paddle2onnx --model_dir inference/ch_PP-OCRv4-teacher-2025-3-28 \
            --model_filename inference.json \
            --params_filename inference.pdiparams \
            --save_file inference-textloc-mulcls-3-31.onnx \
            --enable_onnx_checker True


python3 tools/infer/predict_det.py --image_dir ./inference_results \
                                   --det_model_dir ./weights/hou-inference-textloc-mulcls.onnx \
                                   --use_gpu True \
                                   --use_onnx True