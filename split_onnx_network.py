import os
import onnx
import numpy
import argparse
import onnxruntime




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('onnx_file', type=str, help='config path')
    parser.add_argument('--input-name', '-it', required=True,
                        type=str, help='input tensor name ie. -it images')
    parser.add_argument('--inter-tensors', '-tt', required=True, type=str,
                        help='intermediate tensor to split the network,  ie. -tt 370,425,480,535')
    parser.add_argument('--outputs-name','-ot', required=True,
                        default='output', type=str, help='output tensor name, ie. -ot output')
    parser.add_argument('--output_dir', '-o', type=str, default='./onnx_models/', help='onnx output file path')
    opt = parser.parse_args()
    print(opt)

    input_file_path = opt.onnx_file
    input_names = [opt.input_name]
    output_cnn_names = [opt.outputs_name]

    # tensor_intermediate_outputs = ["370", "425", "480", "535"]
    tensor_intermediate_outputs = opt.inter_tensors.split(",")
    optimized_of_path = os.path.join(
        os.path.dirname(input_file_path),
        os.path.basename(input_file_path).split('.')[0] + str('.optimized.onnx'))
    onnx.utils.extract_model(input_file_path, str(optimized_of_path), input_names, tensor_intermediate_outputs)

    postproc_of_path = os.path.join(
        os.path.dirname(input_file_path),
        os.path.basename(input_file_path).split('.')[0] + str('.postproc.onnx'))
    onnx.utils.extract_model(input_file_path, str(postproc_of_path), tensor_intermediate_outputs, output_cnn_names)

    # Checks
    onnx_model = onnx.load(optimized_of_path)  # load onnx model
    onnx.checker.check_model(onnx_model)
    print('complete ONNX model')
    print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model

    onnx_model = onnx.load(postproc_of_path)  # load onnx model
    onnx.checker.check_model(onnx_model)  # check onnx model
    print('complete ONNX extracted model')
    print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model

    print('ONNX optimized saved as %s' % optimized_of_path)
    print('ONNX postproc  saved as %s' % postproc_of_path)
