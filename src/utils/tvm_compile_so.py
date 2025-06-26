import tvm
from tvm import relay
from tvm.relay.frontend import from_tflite
from tvm.relay.backend import Executor
from tvm.relay.backend import Runtime
from tvm.runtime import save_param_dict
import tflite_runtime.interpreter as tflite
from tvm.micro import export_model_library_format
from tflite import Model
import os


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


def compile_model(mod, params, target="llvm", compile_to="so"):
    if compile_to=="so":
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, params=params)
    elif compile_to=="c":
        with tvm.transform.PassContext(opt_level=3, config={
            "tir.disable_vectorize": True,
            "relay.FuseOps.link_params": True
        }):
            lib = relay.build(
                mod,
                target=target,
                executor=Executor("aot", {
                    "interface-api": "c", 
                    "unpacked-api": True
                    }),
                runtime=Runtime("crt"),
                params=params,
            )
    return lib


def save_artifacts(lib, output_prefix="tvm", compile_to="so"):
    if compile_to == "so":
        lib.export_library(f"{output_prefix}_model.so")

        with open(f"{output_prefix}_params.params", "wb") as f:
            f.write(save_param_dict(lib.get_params()))
        with open(f"{output_prefix}_graph.json", "w") as f:
            f.write(lib.get_graph_json())

    elif compile_to == "c":
        export_model_library_format(lib, "model-micro.tar")


def main():
    model_path = "runs/trained_models/train_0613.tflite"
    input_shape = [1, 2, 32, 10]
    input_dtype = "float32"
    compile_to = "c"

    if compile_to == "so":
        target = "llvm"
    elif compile_to == "c":
        target = tvm.target.Target("c -keys=cpu")

    tflite_model = load_tflite_model(model_path)
    input_name = get_input_details(model_path)
    mod, params = convert_to_relay(tflite_model, input_name, input_shape, input_dtype)

    print("Relay IR module:")
    print(mod)

    lib = compile_model(mod, params, target=target, compile_to=compile_to)
    save_artifacts(lib, output_prefix="tvm", compile_to=compile_to)


if __name__ == "__main__":
    main()
