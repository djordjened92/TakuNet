import os
import numpy as np
import onnx
from PIL import Image
from hailo_sdk_client import ClientRunner, InferenceContext

from hailo_platform import (
    HEF,
    ConfigureParams,
    FormatType,
    HailoSchedulingAlgorithm,
    HailoStreamInterface,
    InferVStreams,
    InputVStreamParams,
    InputVStreams,
    OutputVStreamParams,
    OutputVStreams,
    VDevice,
)

def convert_onnx_to_har(workdir, onnx_path, hw_arch):
    print(f"Converting ONNX model {onnx_path} to HAR for {hw_arch} architecture.")
    onnx_model_name = os.path.basename(onnx_path).split(".")[0]

    onnx_model = onnx.load(onnx_path)
    input_dims = onnx_model.graph.input[0].type.tensor_type.shape.dim[1:]
    input_shape = [1] + [dim.dim_value for dim in input_dims]

    input_shape = [1, 3, 224, 224] if input_shape == 0 else input_shape

    runner = ClientRunner(hw_arch=hw_arch)
    hn, npz = runner.translate_onnx_model(
        onnx_path,
        onnx_model_name,
        start_node_names=["input"],
        end_node_names=["output"],
        net_input_shapes={"input": input_shape},
    )

    hailo_model_har_name = f"{onnx_model_name}_hailo.har"
    if not os.path.exists(workdir):
        os.makedirs(workdir)
    runner.save_har(os.path.join(workdir, hailo_model_har_name))

    print(f"Converted HAR model saved to {workdir}/{hailo_model_har_name}")

def find_har(workdir, original_model_path):
    models = [f for f in os.listdir(workdir) if f.endswith('.har')]

    for model in models:
        if os.path.basename(original_model_path).split(".")[0] in model.split(".")[0]:
            assert os.path.isfile(os.path.join(workdir, model)), f"Model {model} not found in {workdir}"
            model_path = os.path.join(workdir, model)
            return model_path, model.split(".")[0]

    return None, None

def get_calib_dataset(dataset_size, data_path, target_size, workdir):
    output_file = os.path.join(workdir, "calib_set.npy")

    if os.path.exists(output_file):
        print(f"Loading calibration dataset from {output_file}")
        calib_dataset = np.load(output_file)

        return calib_dataset
    
    print(f"Creating calibration dataset from {data_path}")
    calib_dataset = []
    original_images = []

    subdirs = os.listdir(data_path)
    for subdir in subdirs:
        subdir_path = os.path.join(data_path, subdir)
        if os.path.isdir(subdir_path):
            images = [f for f in os.listdir(subdir_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
            original_images.extend([os.path.join(subdir_path, img) for img in images])

    if len(original_images) == 0:
        raise ValueError("No images found in the specified dataset directory.")
    
    original_images = original_images[:dataset_size]
    np.random.shuffle(original_images)

    for image_path in original_images:
        image = Image.open(image_path).convert('RGB')
        image = image.resize(target_size)
        image = np.array(image).astype(np.float32) / 255.0
        calib_dataset.append(image)
    
    np.save(output_file, calib_dataset)
    return np.array(calib_dataset)

