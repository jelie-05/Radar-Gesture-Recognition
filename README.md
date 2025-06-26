# Radar-Gesture-Recognition
![PyDino Game](https://github.com/user-attachments/assets/b689bdab-9586-4f93-a01f-4e100f59c87e)
## Content

## Radar Gesture Recognition

### Background

This project demonstrates gesture recognition using the `Infineon BGT60TR13C FMCW radar`, implemented for a custom TPU board. It focuses on simple gestures such as **up**, **down**, and **hold**, which are used to control a PyGame Dino game in real time. The goal is to showcase real-time embedded inference with minimal latency.

<p align="center">
  <img src="https://github.com/user-attachments/assets/d3199fe0-19bf-438e-93f2-f562a4b02f17" alt="Radar" width="250">
</p>

The main input is a **Range-Doppler map**, which is projected into the time domain to capture temporal features. This projection enables efficient temporal learning while reducing input dimensionality. An additional range-angle map projection has also been implemented, but it currently introduces latency and affects real-time performance.

---

### Dataset Generation

**Data Collection**

To collect gesture data, run the following script:

```bash
python src/utils/rawdata_collect.py
```
Before starting the recording, set the appropriate parameters in the script: `
- `self.recording_type`: specify the target gesture class (e.g., 'push', 'pull', 'hold', 'nothing')
- `self.num_frames`: define the number of frames to record per sequence

Furthermore, ensure that the `time_per_frame` setting used during data collection matches the one used during inference, to maintain consistent temporal resolution across the pipeline. 

Gestures can be performed repeatedly during a single recording session. The raw data format is stored as numpy array to ensure flexibility of the data usage in the future. The data will be annotated later.

**Annotation**

To annotate the recorded data, use the annotation tool:
```bash
python src/utils/annotation.py
```
This script allows you to label and automatically store individual gesture segments in the collected recordings as CSV file. The annotation consists of `file_name`, `gesture`, `start_frame` defining where the gesture starts, and number of samples in the recording.

---

### Train & Inference

A simple CNN model is used to allow efficient deployment on the custom TPU board.

**Training**  
To train the model on annotated radar data, run:

```bash
python src/train.py
