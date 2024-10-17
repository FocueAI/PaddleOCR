# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tqdm import tqdm
import os
import sys
import json

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import paddle

from ppocr.data import create_operators, transform
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model
from ppocr.utils.utility import get_image_file_list
import tools.program as program


def main():
    global_config = config['Global']

    # build post process
    post_process_class = build_post_process(config['PostProcess'],
                                            global_config)

    # build model
    if hasattr(post_process_class, 'character'):
        char_num = len(getattr(post_process_class, 'character'))
        if config["Architecture"]["algorithm"] in ["Distillation",
                                                   ]:  # distillation model
            for key in config["Architecture"]["Models"]:
                if config["Architecture"]["Models"][key]["Head"][
                        "name"] == 'MultiHead':  # multi head
                    out_channels_list = {}
                    if config['PostProcess'][
                            'name'] == 'DistillationSARLabelDecode':
                        char_num = char_num - 2
                    if config['PostProcess'][
                            'name'] == 'DistillationNRTRLabelDecode':
                        char_num = char_num - 3
                    out_channels_list['CTCLabelDecode'] = char_num
                    out_channels_list['SARLabelDecode'] = char_num + 2
                    out_channels_list['NRTRLabelDecode'] = char_num + 3
                    config['Architecture']['Models'][key]['Head'][
                        'out_channels_list'] = out_channels_list
                else:
                    config["Architecture"]["Models"][key]["Head"][
                        "out_channels"] = char_num
        elif config['Architecture']['Head'][
                'name'] == 'MultiHead':  # multi head
            out_channels_list = {}
            char_num = len(getattr(post_process_class, 'character'))
            if config['PostProcess']['name'] == 'SARLabelDecode':
                char_num = char_num - 2
            if config['PostProcess']['name'] == 'NRTRLabelDecode':
                char_num = char_num - 3
            out_channels_list['CTCLabelDecode'] = char_num
            out_channels_list['SARLabelDecode'] = char_num + 2
            out_channels_list['NRTRLabelDecode'] = char_num + 3
            config['Architecture']['Head'][
                'out_channels_list'] = out_channels_list
        else:  # base rec model
            config["Architecture"]["Head"]["out_channels"] = char_num
    model = build_model(config['Architecture'])

    load_model(config, model)

    # create data ops
    transforms = []
    for op in config['Eval']['dataset']['transforms']:
        op_name = list(op)[0]
        if 'Label' in op_name:
            continue
        elif op_name in ['RecResizeImg']:
            op[op_name]['infer_mode'] = True
        elif op_name == 'KeepKeys':
            if config['Architecture']['algorithm'] == "SRN":
                op[op_name]['keep_keys'] = [
                    'image', 'encoder_word_pos', 'gsrm_word_pos',
                    'gsrm_slf_attn_bias1', 'gsrm_slf_attn_bias2'
                ]
            elif config['Architecture']['algorithm'] == "SAR":
                op[op_name]['keep_keys'] = ['image', 'valid_ratio']
            elif config['Architecture']['algorithm'] == "RobustScanner":
                op[op_name][
                    'keep_keys'] = ['image', 'valid_ratio', 'word_positons']
            else:
                op[op_name]['keep_keys'] = ['image']
        transforms.append(op)
    global_config['infer_mode'] = True
    ops = create_operators(transforms, global_config)

    save_res_path = config['Global'].get('save_res_path',
                                         "./output/rec/predicts_rec.txt")
    if not os.path.exists(os.path.dirname(save_res_path)):
        os.makedirs(os.path.dirname(save_res_path))

    model.eval()
    
    infer_imgs = config['Global']['infer_img']
    infer_list = config['Global'].get('infer_list', None)
    ############## ----------------------------- 重写paddleOCR 推理逻辑 ----------------------------------###################
    ######################################## 临时测试准确率 设置 ############################################################

    # label_dir = r'D:\projects\PaddleOCR\train_data\rec\tot_source'
    label_dir = r'/usr/projects/new_paddleOCR_2024_9_9/PaddleOCR/train_data/rec/test'
    count = 0
    accuracy_cnt = 0
    
    ########################################################################################################################
    for file in tqdm(os.listdir(infer_imgs)):
        filename_without_ext, ext = os.path.splitext(file)
        if ext in ['.jpg','.png']:
            detail_img_path = os.path.join(infer_imgs,file)
            detail_label_path = os.path.join(label_dir,filename_without_ext+'.txt')
            assert os.path.exists(detail_label_path)
            with open(detail_label_path, 'r', encoding='utf-8') as reader:
                label = reader.readline()

            with open(detail_img_path, 'rb') as f:
                img = f.read()
                data = {'image': img}
            batch = transform(data, ops)
            images = np.expand_dims(batch[0], axis=0)
            images = paddle.to_tensor(images)
            preds = model(images)
            post_result = post_process_class(preds)
            # print(f'post_result:{post_result}')
            pred_label, pred_score = post_result[0][0], post_result[0][1]
            count += 1
            # print(f'label:{label}, pred:{pred_label}, score:{pred_score}')
            if ' ' in pred_label:
                print(f'with_blank_pred:{pred_label}')
            if label.strip().replace(' ','').replace('（','(').replace('）',')').replace('：',':') == pred_label.strip().replace(' ','').replace('（','(').replace('）',')').replace('：',':') :
                accuracy_cnt +=1
                 
            else:
                pass
                # print('-'*10)
                # print(f'label:{label}, pred:{pred_label}, score:{pred_score}')
            print(f'accuracy:{accuracy_cnt/count}')
    print(f'accuracy:{accuracy_cnt/count}')


    ############## ----------------------------- 重写paddleOCR 推理逻辑 ----------------------------------###################
    # with open(save_res_path, "w") as fout:
    #     for file in get_image_file_list(infer_imgs, infer_list=infer_list):
    #         logger.info("infer_img: {}".format(file))
    #         with open(file, 'rb') as f:
    #             img = f.read()
    #             data = {'image': img}
    #         batch = transform(data, ops)
    #         if config['Architecture']['algorithm'] == "SRN":
    #             encoder_word_pos_list = np.expand_dims(batch[1], axis=0)
    #             gsrm_word_pos_list = np.expand_dims(batch[2], axis=0)
    #             gsrm_slf_attn_bias1_list = np.expand_dims(batch[3], axis=0)
    #             gsrm_slf_attn_bias2_list = np.expand_dims(batch[4], axis=0)

    #             others = [
    #                 paddle.to_tensor(encoder_word_pos_list),
    #                 paddle.to_tensor(gsrm_word_pos_list),
    #                 paddle.to_tensor(gsrm_slf_attn_bias1_list),
    #                 paddle.to_tensor(gsrm_slf_attn_bias2_list)
    #             ]
    #         if config['Architecture']['algorithm'] == "SAR":
    #             valid_ratio = np.expand_dims(batch[-1], axis=0)
    #             img_metas = [paddle.to_tensor(valid_ratio)]
    #         if config['Architecture']['algorithm'] == "RobustScanner":
    #             valid_ratio = np.expand_dims(batch[1], axis=0)
    #             word_positons = np.expand_dims(batch[2], axis=0)
    #             img_metas = [
    #                 paddle.to_tensor(valid_ratio),
    #                 paddle.to_tensor(word_positons),
    #             ]
    #         if config['Architecture']['algorithm'] == "CAN":
    #             image_mask = paddle.ones(
    #                 (np.expand_dims(
    #                     batch[0], axis=0).shape), dtype='float32')
    #             label = paddle.ones((1, 36), dtype='int64')
    #         images = np.expand_dims(batch[0], axis=0)
    #         images = paddle.to_tensor(images)
    #         if config['Architecture']['algorithm'] == "SRN":
    #             preds = model(images, others)
    #         elif config['Architecture']['algorithm'] == "SAR":
    #             preds = model(images, img_metas)
    #         elif config['Architecture']['algorithm'] == "RobustScanner":
    #             preds = model(images, img_metas)
    #         elif config['Architecture']['algorithm'] == "CAN":
    #             preds = model([images, image_mask, label])
    #         else:
    #             preds = model(images)
    #         post_result = post_process_class(preds)
    #         info = None
    #         if isinstance(post_result, dict):
    #             rec_info = dict()
    #             for key in post_result:
    #                 if len(post_result[key][0]) >= 2:
    #                     rec_info[key] = {
    #                         "label": post_result[key][0][0],
    #                         "score": float(post_result[key][0][1]),
    #                     }
    #             info = json.dumps(rec_info, ensure_ascii=False)
    #         elif isinstance(post_result, list) and isinstance(post_result[0],
    #                                                           int):
    #             # for RFLearning CNT branch 
    #             info = str(post_result[0])
    #         else:
    #             if len(post_result[0]) >= 2:
    #                 info = post_result[0][0] + "\t" + str(post_result[0][1])

    #         if info is not None:
    #             logger.info("\t result: {}".format(info))
    #             fout.write(file + "\t" + info + "\n")
    # logger.info("success!")


if __name__ == '__main__':
    config, device, logger, vdl_writer = program.preprocess()
    main()
