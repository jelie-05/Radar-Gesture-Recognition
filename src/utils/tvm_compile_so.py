import tvm
from tvm import relay
from tvm.relay.frontend import from_tflite
from tvm.runtime import save_param_dict
import tflite_runtime.interpreter as tflite
from tflite import Model


def load_tflite_model(model_path: str):
    with open(model_path, "rb") as f:
        model_buf = f.read()
    return Model.GetRootAsModel(model_buf, 0)


def get_input_details(model_path: str):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    input_name = input_details[0]["name"]
    return input_name


def convert_to_relay(tflite_model, input_name: str, input_shape, input_dtype):
    shape_dict = {input_name: input_shape}
    dtype_dict = {input_name: input_dtype}
    mod, params = from_tflite(tflite_model, shape_dict=shape_dict, dtype_dict=dtype_dict)
    return mod, params


def compile_model(mod, params, target="llvm"):
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)
    return lib


def save_artifacts(lib, output_prefix="tvm"):
    lib.export_library(f"{output_prefix}_model.so")
    with open(f"{output_prefix}_params.params", "wb") as f:
        f.write(save_param_dict(lib.get_params()))
    with open(f"{output_prefix}_graph.json", "w") as f:
        f.write(lib.get_graph_json())


def main():
    model_path = "runs/trained_models/train_0613.tflite"
    input_shape = [1, 2, 32, 10]
    input_dtype = "float32"

    tflite_model = load_tflite_model(model_path)
    input_name = get_input_details(model_path)
    mod, params = convert_to_relay(tflite_model, input_name, input_shape, input_dtype)

    print("Relay IR module:")
    print(mod)

    lib = compile_model(mod, params, target="llvm")
    save_artifacts(lib, output_prefix="tvm")


if __name__ == "__main__":
    main()
