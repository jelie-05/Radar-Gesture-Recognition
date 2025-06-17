import torch
from src.model.simple_model import SimpleCNN
import os
from onnx_tf.backend import prepare
import onnx
import tensorflow as tf
import numpy as np

# === 1. Load the PyTorch model ===
model = SimpleCNN(in_channels=2, num_classes=4)
date = '0613'
model_path = f'runs/trained_models/train_{date}-last.pth'

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    print("[Model] Loaded successfully.")
    input("[Model] Press Enter to continue...")
else:
    input("[Model] Model file not found. Please ensure the path is correct and the model is trained.")
    model = model.to('cpu')

model.eval()

# === 2. Export to ONNX ===
dummy_input = torch.randn(1, 2, 32, 10)
onnx_path = f'runs/trained_models/train_{date}.onnx'

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: "batch_size"}, 'output': {0: "batch_size"}},
    export_params=True,
    opset_version=11,
    do_constant_folding=True
)
print(f"[ONNX] Exported model to {onnx_path}")

# === 3. Convert ONNX to TensorFlow SavedModel ===
onnx_model = onnx.load(onnx_path)
tf_rep = prepare(onnx_model)
saved_model_dir = "saved_model"
tf_rep.export_graph(saved_model_dir)
print(f"[TF] SavedModel exported to {saved_model_dir}")

# === 4. Convert TensorFlow SavedModel to TFLite ===
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

tflite_path = f'runs/trained_models/train_{date}.tflite'
with open(tflite_path, "wb") as f:
    f.write(tflite_model)
print(f"[TFLite] Model saved to {tflite_path}")

# === 5. Load and Test the TFLite model ===
try:
    import tflite_runtime.interpreter as tflite
    print("[TFLite] Using tflite_runtime.Interpreter")
except ImportError:
    print("[TFLite] Falling back to tf.lite.Interpreter")
    tflite = tf.lite

interpreter = tflite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Check input shape and dtype
input_shape = input_details[0]['shape']
input_dtype = input_details[0]['dtype']
print(f"[TFLite] Input shape: {input_shape}, dtype: {input_dtype}")

# Create dummy input
dummy_input_np = np.random.rand(*input_shape).astype(input_dtype)

# Run inference
interpreter.set_tensor(input_details[0]['index'], dummy_input_np)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

# Display result
print("[TFLite] Inference successful!")
print("Output shape:", output_data.shape)
print("Output (first sample):", output_data[0] if output_data.ndim > 1 else output_data)
