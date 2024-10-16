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
















