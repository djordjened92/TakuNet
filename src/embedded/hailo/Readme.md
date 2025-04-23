# How to Run TakuNet on Hailo8 Accelerators

> **Note**: This guide provides supplementary information and code to enable support for Hailo8 accelerators. It complements the TakuNet publication at WACV2025.

> **Warning**: The code has been tested with:
> - Hailo 4.21 on the workstation used to generate the `.Hef` file.
> - Hailo 4.20 on the embedded device.

---

## Prerequisites

1. **Register and Login**  
    - Visit [hailo.ai](https://hailo.ai) and log in.
    - Navigate to the "Developer Zone" and download `hailo_ai_sw_suite_2025-04`.

2. **Install Docker**  
    - Follow the instructions provided in this [AutoDock repository](https://github.com/DanielRossi1/AutoDock).

---

## Steps to Run TakuNet on Hailo8

### 1. Prepare the Hailo Software Suite
- Extract the downloaded `.zip` file to obtain:
  - A `.tar.gz` file (Docker container).
  - An executable `.sh` file to build and run the container.
- Run the `.sh` file and wait for the container to build.

---

### 2. Configure TakuNet
- Open the `configs/TakuNet_datasetname.yml` file and make the following changes:
  ```yaml
  mode: export
  onnx_opset_version: 15  # Or choose your preferred version
  ```
- Launch TakuNet:
  ```bash
  ./launch TakuNet_datasetname
  ```
- Copy the exported ONNX weights into the `shared_with_docker` folder created during the container build:
  ```bash
  cd shared_with_docker
  mkdir -p models
  cp path/to/exports/TakuNet_fp16_opsetXX.onnx /path/to/shared_with_docker/TakuNet_fp16_opsetXX.onnx
  ```

---

### 3. Prepare the Embedded Files
- From the `embedded/hailo` directory, copy the following files into the `shared_with_docker` folder:
  - `main.py`
  - `infer_hailo`
  - `hailo.yml`
- Create a `datasets` folder inside `shared_with_docker` and copy the dataset into this folder. The dataset will be used during the calibration phase:
  ```bash
  cd shared_with_docker
  mkdir -p datasets
  cp -r /path/to/dataset ./datasets/
  ```

---

### 4. Run the Application
- Execute the following command:
  ```bash
  python3 main.py --cfg-path hailo.yml
  ```

---

Enjoy running TakuNet on Hailo8 Accelerators!