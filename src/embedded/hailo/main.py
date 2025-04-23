import argparse
import os

import yaml

from infer_hailo import get_calib_dataset, find_har, convert_onnx_to_har
from hailo_sdk_client import ClientRunner, InferenceContext

def infer_hailo(settings: dict, workdir: str) -> None:
    if not os.path.isfile(settings["onnx_model_path"].split(".")[0] + "_hailo.har"):
        convert_onnx_to_har(workdir, settings["onnx_model_path"], settings["hailo_version"])
    else:
        print(f"Model {settings['onnx_model_path']} already converted to HAR.")

    model_path, model_name = find_har(workdir, settings["onnx_model_path"])

    if not model_path or not model_name:
        raise ValueError("HAR model not found.")

    runner = ClientRunner(har=model_path, hw_arch=settings["hailo_version"])
    calib_dataset = get_calib_dataset(settings["dataset_size"], settings["data_path"], settings["target_size"], workdir)

    print("Calibrating the model...")
    if not os.path.isfile(os.path.join(workdir, f"{model_name}_quantized_model.har")):
        with runner.infer_context(InferenceContext.SDK_NATIVE) as ctx:
            runner.optimize(calib_dataset)
            quantized_model_har_path = f"{model_name}_quantized_model.har"
            runner.save_har(os.path.join(workdir, quantized_model_har_path))
            print(f"Quantized HAR model saved to {os.path.join(workdir, quantized_model_har_path)}")
    else:
        print(f"Quantized model already exists at {os.path.join(workdir, f'{model_name}_quantized_model.har')}")
        runner = ClientRunner(har=os.path.join(workdir, f'{model_name}_quantized_model.har'), hw_arch=settings["hailo_version"])

    print("Running inference...")
    with runner.infer_context(InferenceContext.SDK_NATIVE) as ctx:
        hef = runner.compile()
        file_name = f"{model_name}.hef"
        with open(os.path.join(workdir, file_name), "wb") as f:
            f.write(hef)
        print(f"Compiled HEF file saved to {os.path.join(workdir, file_name)}")

def load_config(cfg_path):
    with open(cfg_path, 'r') as f:
        settings = yaml.safe_load(f)
    return settings

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument("--cfg-path", type=str, required=True, help="Path to the configuration file")

    parser.add_argument("--onnx-model-path", type=str, help="Path to the ONNX model")

    parser.add_argument("--dataset-size", type=int, default=1000, help="Size of the dataset for inference")
    parser.add_argument("--data-path", type=str, help="Path to the dataset")
    parser.add_argument("--hailo-version", type=str, default="hailo8", help="Hailo version")
    parser.add_argument("--target-size", type=int, nargs=2, help="Target size for the images")

    args = parser.parse_args()
    settings = load_config(args.cfg_path)
    settings["target_size"] = tuple(settings["target_size"])

    print("Settings loaded from config file:")
    print(settings)

    infer_hailo(settings, os.path.dirname(settings["onnx_model_path"]))
