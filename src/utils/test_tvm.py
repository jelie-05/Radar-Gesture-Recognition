import tvm
from tvm import relay, transform
from tvm.relay.frontend import from_tflite
from tvm.runtime import save_param_dict
import tflite_runtime.interpreter as tflite
from tflite import Model

# Load the TFLite model file
model_path = "runs/trained_models/train_0613.tflite"
with open(model_path, "rb") as f:
    model_buf = f.read()

tflite_model = Model.GetRootAsModel(model_buf, 0)

# Detect the correct input tensor name from interpreter
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
input_name = input_details[0]['name']  # => "serving_default_input:0"

# Define correct shape and dtype
shape_dict = {input_name: [1, 2, 32, 10]}
dtype_dict = {input_name: "float32"}

# Convert to TVM Relay IR
mod, params = from_tflite(tflite_model, shape_dict=shape_dict, dtype_dict=dtype_dict)

# Show converted graph
print(mod)

# Choose the target
target = "llvm"

# Optimization pass
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

# Save the compiled library; native machine code
lib.export_library("tvm_model.so")

# Save the parameters to binary
with open("tvm_params.params", "wb") as f:
    f.write(save_param_dict(lib.get_params()))

# Save graph computation (JSON)
with open("graph.json", "w") as f:
    f.write(lib.get_graph_json())