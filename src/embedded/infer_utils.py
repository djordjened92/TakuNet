import os
import numpy as np
import subprocess

def calculate_fps(elapsed_times):
    avg_time = np.mean(elapsed_times)
    fps = 1.0 / avg_time if avg_time > 0 else 0
    return fps, avg_time

def try_export_trt_engine(command: list):
    try:
        print(f"Exporting TensorRT engine with command: {command}")
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Output: {result.stdout.decode()}")
        print(f"Error: {result.stderr.decode()}")
    except subprocess.CalledProcessError as e:
        print(f"Error during trtexec execution: {e}")
        return False
    return True

def export_trt_engine(settings):
    onnx_model_path = settings["onnx_model_path"]

    assert os.path.exists(onnx_model_path), "Model path does not exist!"
    assert onnx_model_path.endswith(".onnx"), "Model path must be an .onnx file!"

    engine_model_path = onnx_model_path.replace(".onnx", ".engine")

    command = [
        "/usr/src/tensorrt/bin/trtexec",
        "--skipInference",
        "--fp16",
        "--useDLACore=0",
        "--allowGPUFallback",
        "--timingCacheFile=timing.cache",
        "--shapes=input:1x3x240x240",
        "--precisionConstraints=prefer",
        "--layerPrecisions=*:fp16",
        "--layerOutputTypes=*:fp16",
        "--inputIOFormats=fp16:chw",
        "--outputIOFormats=fp16:chw",
        f"--onnx={onnx_model_path}", 
        f"--saveEngine={engine_model_path}",
        "--verbose",
    ]
    if try_export_trt_engine(command):
        return
    
    print("Retrying with different settings...")
    command = [
        "/usr/src/tensorrt/bin/trtexec",
        "--skipInference",
        "--fp16",
        "--timingCacheFile=timing.cache",
        "--shapes=input:1x3x240x240",
        "--precisionConstraints=prefer",
        "--layerPrecisions=*:fp16",
        "--layerOutputTypes=*:fp16",
        "--inputIOFormats=fp16:chw",
        "--outputIOFormats=fp16:chw",
        f"--onnx={onnx_model_path}", 
        f"--saveEngine={engine_model_path}",
    ]

    if try_export_trt_engine(command):
        return
    
    print("Retrying with different settings...")
    command = [
        "/usr/src/tensorrt/bin/trtexec",
        "--skipInference",
        "--fp16",
        "--shapes=input:1x3x240x240",
        f"--onnx={onnx_model_path}", 
        f"--saveEngine={engine_model_path}",
    ]

    if try_export_trt_engine(command):
        return

    print("Retrying with different settings...")
    command = [
        "/usr/src/tensorrt/bin/trtexec",
        "--buildOnly",
        "--fp16",
        "--shapes=input:1x3x240x240",
        f"--onnx={onnx_model_path}", 
        f"--saveEngine={engine_model_path}",
    ]
    if try_export_trt_engine(command):
        return
    else:
        print("Could not export TensorRT engine automatically!")
        print("Please run the command manually, be sure to place the .engine file within /home/user/src/exports or change .engine path in embedded/config/TakuNet.yml")
        exit(1)
    