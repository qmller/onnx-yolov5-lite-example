# Copyright 2023 The Kalray Authors. All Rights Reserved.
#
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy
import kann


def onnx_resize_parser_callback(neural_network, prev_imgs, onnx_node, model_info):
    roi = prev_imgs[1]
    if isinstance(roi, numpy.ndarray) and len(roi) == 0:
        prev_imgs[1] = None
    return kann.layers.resize.onnx_parser_callback(
        neural_network, prev_imgs, onnx_node, model_info)

onnx_parser_callbacks = {
    'Resize': onnx_resize_parser_callback,
}
