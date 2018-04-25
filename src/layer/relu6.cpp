// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "relu6.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(ReLU6)

ReLU6::ReLU6()
{
    one_blob_only = true;
    support_inplace = true;
}

int ReLU6::load_param(const ParamDict& pd)
{
    slope = pd.get(0, 0.f);

    return 0;
}

int ReLU6::forward_inplace(Mat& bottom_top_blob) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;

    if (slope == 0.f)
    {
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                if (ptr[i] < 0)
                    ptr[i] = 0;
                if (ptr[i] > 6)
                    ptr[i] = 6;
            }
        }
    }
    else
    {
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                if (ptr[i] < 0)
                    ptr[i] *= slope;
                if (ptr[i] > 6)
                    ptr[i] = 6;
            }
        }
    }

    return 0;
}

} // namespace ncnn
