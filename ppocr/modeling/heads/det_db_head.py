# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import paddle
from paddle import nn
import paddle.nn.functional as F
from paddle import ParamAttr
from ppocr.modeling.backbones.det_mobilenet_v3 import ConvBNLayer


def get_bias_attr(k):
    stdv = 1.0 / math.sqrt(k * 1.0)
    initializer = paddle.nn.initializer.Uniform(-stdv, stdv)
    bias_attr = ParamAttr(initializer=initializer)
    return bias_attr


class Head(nn.Layer):
    def __init__(self, in_channels, kernel_list=[3, 2, 2], **kwargs):
        super(Head, self).__init__()
        # TODO: 增加 多分类 功能!!!
        self.mulcls = kwargs.get("n_cls", None)
        self.conv1 = nn.Conv2D(
            in_channels=in_channels,
            out_channels=in_channels // 4,
            kernel_size=kernel_list[0],
            padding=int(kernel_list[0] // 2),
            weight_attr=ParamAttr(),
            bias_attr=False,
        )
        self.conv_bn1 = nn.BatchNorm(
            num_channels=in_channels // 4,
            param_attr=ParamAttr(initializer=paddle.nn.initializer.Constant(value=1.0)),
            bias_attr=ParamAttr(initializer=paddle.nn.initializer.Constant(value=1e-4)),
            act="relu",
        )

        self.conv2 = nn.Conv2DTranspose(
            in_channels=in_channels // 4,
            out_channels=in_channels // 4,
            kernel_size=kernel_list[1],
            stride=2,
            weight_attr=ParamAttr(initializer=paddle.nn.initializer.KaimingUniform()),
            bias_attr=get_bias_attr(in_channels // 4),
        )
        self.conv_bn2 = nn.BatchNorm(
            num_channels=in_channels // 4,
            param_attr=ParamAttr(initializer=paddle.nn.initializer.Constant(value=1.0)),
            bias_attr=ParamAttr(initializer=paddle.nn.initializer.Constant(value=1e-4)),
            act="relu",
        )
        self.conv3 = nn.Conv2DTranspose(
            in_channels=in_channels // 4,
            out_channels= 1,
            # out_channels= self.mulcls if self.mulcls is not None else 1,
            kernel_size=kernel_list[2],
            stride=2,
            weight_attr=ParamAttr(initializer=paddle.nn.initializer.KaimingUniform()),
            bias_attr=get_bias_attr(in_channels // 4),
        )
        self.con_mul_cls = nn.Conv2DTranspose(
            # in_channels=in_channels // 4,
            in_channels=in_channels // 2,
            out_channels= self.mulcls if self.mulcls is not None else 1,
            kernel_size=kernel_list[2],
            stride=2,
            weight_attr=ParamAttr(initializer=paddle.nn.initializer.KaimingUniform()),
            bias_attr=get_bias_attr(in_channels // 4),
        )
        ###############################################  为强壮分类分支做的代码修改
        self.extra_conv = nn.Conv2D(
            in_channels=in_channels // 4,
            out_channels=in_channels // 2,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(initializer=paddle.nn.initializer.KaimingUniform()),
            bias_attr=get_bias_attr(in_channels // 4)
        )
        self.extra_bn = nn.BatchNorm(
            num_channels=in_channels // 2,
            param_attr=ParamAttr(initializer=paddle.nn.initializer.Constant(value=1.0)),
            bias_attr=ParamAttr(initializer=paddle.nn.initializer.Constant(value=1e-4)),
            act="relu"
        )
        self.dropout = nn.Dropout(p=0.5)
        #######################################################

    def forward(self, x, return_f=False, only_return_mulcls=False):  # x.shape = [1, 256, 24, 320]
        x = self.conv1(x)     # [1, 64, 24, 320]   通道数下降 4 倍
        x = self.conv_bn1(x)  # [1, 64, 24, 320]    
        x = self.conv2(x)     # [1, 64, 48, 640]    特征图变为原来的 2 倍
        x = self.conv_bn2(x)  # [1, 64, 48, 640]
        if return_f is True:
            f = x
        if only_return_mulcls and self.mulcls is not None:   # x.shape = [1, 64, 48, 640] 
            # TODO: 准备在这里增加神经元的数量，提高分类的效果！！！！
            # 在这里增加额外的卷积层和激活函数
            x = self.extra_conv(x)  # 增加额外的卷积层   x.shape = [1, 128, 48, 640]
            x = self.extra_bn(x)    # 批量归一化 
            x = self.dropout(x)     # 添加 Dropout  x = [1, 128, 48, 640]
             # 最终的分类卷积层
            
            
            
            
            
            
            return  self.con_mul_cls(x)
        x = self.conv3(x)
        # if self.mulcls is not None:
        #     return x
        x = F.sigmoid(x)
        if return_f is True:
            return x, f
        return x


class DBHead(nn.Layer):
    """
    Differentiable Binarization (DB) for text detection:
        see https://arxiv.org/abs/1911.08947
    args:
        params(dict): super parameters for build DB network
    """

    def __init__(self, in_channels, k=50, **kwargs):
        super(DBHead, self).__init__()
        
        ################################### 文本属性多分类 ################################
        self.mulcls_num = kwargs.get("n_cls", None) 
        
        #################################################################################
        
        
        self.k = k
        self.binarize = Head(in_channels, **kwargs)
        self.thresh = Head(in_channels, **kwargs)
        self.mulcls = Head(in_channels, **kwargs)
        

    def step_function(self, x, y):
        return paddle.reciprocal(1 + paddle.exp(-self.k * (x - y)))

    def forward(self, x, targets=None):
        shrink_maps = self.binarize(x)   # TODO: 我判断这里好像 永远不会执行!!!!
        if not self.training:
            return {"maps": shrink_maps}

        threshold_maps = self.thresh(x)
        binary_maps = self.step_function(shrink_maps, threshold_maps)
        y = paddle.concat([shrink_maps, threshold_maps, binary_maps], axis=1)
        return {"maps": y}


class LocalModule(nn.Layer):
    def __init__(self, in_c, mid_c, use_distance=True):
        super(self.__class__, self).__init__()
        self.last_3 = ConvBNLayer(in_c + 1, mid_c, 3, 1, 1, act="relu")
        self.last_1 = nn.Conv2D(mid_c, 1, 1, 1, 0)

    def forward(self, x, init_map, distance_map):
        outf = paddle.concat([init_map, x], axis=1)
        # last Conv
        out = self.last_1(self.last_3(outf))
        return out


class PFHeadLocal(DBHead):
    def __init__(self, in_channels, k=50, mode="small", **kwargs):
        super(PFHeadLocal, self).__init__(in_channels, k, **kwargs)
        self.mode = mode
        
        self.up_conv = nn.Upsample(scale_factor=2, mode="nearest", align_mode=1)
        if self.mode == "large":
            self.cbn_layer = LocalModule(in_channels // 4, in_channels // 4)
        elif self.mode == "small":
            self.cbn_layer = LocalModule(in_channels // 4, in_channels // 8)

    def forward(self, x, targets=None):  # x.shape=[12, 256, 80, 320]    
        shrink_maps, f = self.binarize(x, return_f=True) # shrink_maps.shape=[12, 1, 320, 1280] ---> 这已经是原图大小了,  f.shape=[12, 64, 160, 640]
        mulcls_feature = self.mulcls(x,only_return_mulcls=True)
        base_maps = shrink_maps
        cbn_maps = self.cbn_layer(self.up_conv(f), shrink_maps, None) # [12, 1, 320, 1280]
        cbn_maps = F.sigmoid(cbn_maps) # [12, 1, 320, 1280]
        if not self.training:
            return {"maps": 0.5 * (base_maps + cbn_maps), "cbn_maps": cbn_maps, "mulcls_feature": mulcls_feature}

        threshold_maps = self.thresh(x)  # [12, 1, 320, 1280]
        binary_maps = self.step_function(shrink_maps, threshold_maps) # shrink_maps.shape=threshold_maps.shape=[12, 1, 320, 1280],,,,,,,binary_maps.shape=[12, 1, 320, 1280]
        y = paddle.concat([cbn_maps, threshold_maps, binary_maps], axis=1)  # y.shape = [12, 3, 320, 1280]
        return {"maps": y, "distance_maps": cbn_maps, "cbn_maps": binary_maps, "mulcls_feature": mulcls_feature}
