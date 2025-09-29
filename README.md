# Radar-Gesture-Recognition
![PyDino Game](https://github.com/user-attachments/assets/b689bdab-9586-4f93-a01f-4e100f59c87e)

## Content
- [Radar Gesture Recognition](#radar-gesture-recognition)
  - [Background](#background)
  - [Dataset Generation](#dataset-generation)
    - [Data Collection](#data-collection)
    - [Annotation](#annotation)
  - [Train & Inference](#train--inference)
- [Deployment on TPU](#deployment-on-tpu)
  - [Converting Trained Model to Runtime](#converting-trained-model-to-runtime)
  - [TVM Compilation](#tvm-compilation)
  - [C++ Deployment](#c-deployment)

## Radar Gesture Recognition

### Background

This project demonstrates gesture recognition using the `Infineon BGT60TR13C FMCW radar`, implemented for a custom TPU board. It focuses on simple gestures such as **up**, **down**, and **hold**, which are used to control a [PyGame Dino game](https://github.com/MaxRohowsky/chrome-dinosaur) in real time. The goal is to showcase real-time embedded inference with minimal latency.

<p align="center">
  <img src="https://github.com/user-attachments/assets/d3199fe0-19bf-438e-93f2-f562a4b02f17" alt="Radar" width="250">
</p>

The main input is a **Range-Doppler map**, which is projected into the time domain to capture temporal features. This projection enables efficient temporal learning while reducing input dimensionality. An additional range-angle map projection has also been implemented, but it currently introduces latency and affects real-time performance.

---

### Dataset Generation

**Data Collection**

To collect gesture data, run the following script:

<!-- ```bash
python src/utils/rawdata_collect.py
``` -->
```bash
python3 -m src.utils.rawdata_collect
```
Before starting the recording, set the appropriate parameters in the script: `
- `self.recording_type`: specify the target gesture class (e.g., 'push', 'pull', 'hold', 'nothing')
- `self.num_frames`: define the number of frames to record per sequence

Furthermore, ensure that the `time_per_frame` setting used during data collection matches the one used during inference, to maintain consistent temporal resolution across the pipeline. 

Gestures can be performed repeatedly during a single recording session. The raw data format is stored as numpy array to ensure flexibility of the data usage in the future. The data will be annotated later.

**Annotation**

To annotate the recorded data, use the annotation tool:
<!-- ```bash
python src/utils/annotation.py
``` -->
```bash
python3 -m src.utils.annotation
```
This script allows you to label and automatically store individual gesture segments in the collected recordings as CSV file. The annotation consists of `file_name`, `gesture`, `start_frame` defining where the gesture starts, and number of samples in the recording.

**Using IFX Dataset**
```
python3 -m src.train_utils.build_dataset.build_transformed \
  --data_dir /home/phd_li/dataset/radar_gesture_dataset/fulldataset/ \
  --output_dir (output directory) \
  --none_class 
  --stepsize 3
```
`none_class`: include none class. `step_size`: use frame every `step_size` to reduce the frequency of the dataset.

---

### Train & Inference

A simple CNN model is used to allow efficient deployment on the custom TPU board.

**Training**  
To train the model on annotated radar data, run from the root:

<!-- ```bash
python src/train.py
``` -->
```bash
python3 -m src.train_distributed --config config/config_file.yaml
```
This script loads the preprocessed dataset, trains the CNN on gesture classes, and saves the resulting model for inference.

**Config File**
`input_channels` corresponds to range-doppler-angle channel. `output_classes` corresponds to number of gestures.

**Inference**

To run real-time inference with radar input and visualize outputs:
<!-- ```bash
python src/realtime_inference.py
``` -->
```bash
python3 -m src.realtime_inference
```
This script handles live radar input and displays both the Range-Time Map (RTM) and Doppler-Time Map (DTM) in real time. It also performs live classification and outputs the predicted gesture.

---

## Deployment on TPU

The trained PyTorch model is converted to ONNX or TFLite format, and then compiled using [Apache TVM](https://github.com/apache/tvm) for deployment on a custom TPU board. The final runtime model is executed using the C++ runtime located in the `cpp_inference/` directory.

---

### Converting Trained Model to Runtime

To convert the PyTorch model to ONNX or TFLite format, change the `run_id` and run:

<!-- ```bash
python src/utils/runtime_convert.py
``` -->
```bash
python3 -m src.utils.runtime_convert
```
This creates an intermediate format suitable for further optimization.

---

### TVM Compilation
We use **TVM v0.13.0** for compiling the model due to its flexibility in targeting multiple hardware platforms. Instead of converting from ONNX, we use TFLite as the input format to TVM for faster deployment after the conversion.

To compile the model using TVM, run:
<!-- ```bash
python src/utils/tvm_transform.py
``` -->
```bash
python3 -m src.utils.tvm_transform
```
You can configure the output format by setting the compile_to argument:

- `so`: produces a device-specific shared object (.so) compiled model (C/C++ runtime).

- `c`: produces a tar archive containing C source code, which can be compiled later for any target (not device-specific).

The resulting .so file or .tar file (extracted) should be placed in `cpp_inference/`.

---

### C++ Deployment

To compile the C++ runtime for the TVM-generated model, navigate to the `build/` directory. Adapt the CMakeList with your extracted folder (e.g., `model-micro`) and run the following:

```bash
cd cpp_inference
mkdir build
cd build
cmake ..
make
```

Depending on the output format from TVM (.so or C source), run the corresponding executable:
- If TVM output is a shared object (.so):
```bash
./run_so
```
- If TVM output is C source code (.c in tar archive):
```bash
./run_c
```
Make sure the compiled artifacts and tvm_model.so or C-generated source are correctly placed in the build directory before running. They should be in `cpp_inference/`.

### Installation Packages

Infineon Radar SDK: Download version ifxAvian 3.3.1 and run:
```
radar_sdk/radar_sdk/libs/linux_x64/
python3 -m pip install ifxAvian-3.3.1-py3-none-any.whl
``` 
