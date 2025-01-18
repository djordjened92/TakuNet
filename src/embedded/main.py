import argparse
import onnxruntime as ort
import torch
from torchvision import datasets, transforms
import timeit
import os
import numpy as np

from tqdm import tqdm
import yaml

from infer import ONNXClassifierWrapper
from infer_oldjetpack import ONNXClassifierWrapper_OldJetpack
from infer_utils import calculate_fps, export_trt_engine

def infer_tensorrt(wrapper, settings, dataset):
    if not os.path.exists(settings['tensorrt_engine_path']):
        print("TensorRT engine not found, exporting...")
        export_trt_engine(settings)

    classifier = wrapper(settings)

    elapsed_times = []
    for i in tqdm(range(settings['dataset_size'])):
        image, _ = dataset[i]
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        image = image.astype(np.float16)

        _, elapsed_time = classifier.predict(image)
        elapsed_times.append(elapsed_time)
    
    return elapsed_times

def infer_onnx(settings, dataset):
    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 0
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
    
    print(providers)
    session = ort.InferenceSession(settings['onnx_model_path'], sess_options=sess_options, providers=providers)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    print(f"ONNXRuntime available devices: {session.get_providers()}")
    print(f"ONNXRuntime current device: {session.get_provider_options()}")
    
    elapsed_times = []
    for i in tqdm(range(settings['dataset_size'])):
        image, _ = dataset[i]
        image = np.expand_dims(image, axis=0)

        image = image.astype(np.float16) if settings["fp16_mode"] else image.astype(np.float32)

        elapsed_time = timeit.timeit(lambda: session.run([output_name], {input_name: image}), number=1)
        elapsed_times.append(elapsed_time)
    
    return elapsed_times

def load_config(cfg_path):
    with open(cfg_path, 'r') as f:
        settings = yaml.safe_load(f)
    return settings

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument("--cfg-path", type=str, required=True, help="Path to the ONNX model")

    args = parser.parse_args()
    settings = load_config(args.cfg_path)

    print(f"Settings: {settings}")
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.FakeData(transform=transform, size=settings['dataset_size'], image_size=settings['img_size'])

    print("Using ONNXClassifierWrapper_OldJetpack") if settings['old_jetpack'] else print("Using ONNXClassifierWrapper")
    ONNXWrapper = ONNXClassifierWrapper_OldJetpack if settings['old_jetpack'] else ONNXClassifierWrapper

    print("Starting inference...")
    if settings['use_tensorrt']:
        elapsed_times = infer_tensorrt(ONNXWrapper, settings, dataset)
    else:
        elapsed_times = infer_onnx(settings, dataset)

    # The first 100 iterations are discarded to avoid the warm-up time
    elapsed_times = elapsed_times[100:]  
    fps, avg_time = calculate_fps(elapsed_times)
    print(f"\nTotal FPS: {fps:.2f}")
    print(f"Average inference latency: {avg_time:.6f} s")