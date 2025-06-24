import tvm
from tvm import relay
from tvm.relay.frontend import from_tflite
from tvm.relay.backend import Executor, Runtime
from tvm.micro import export_model_library_format
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
    input_shape = input_details[0]["shape"]
    input_dtype = input_details[0]["dtype"].__name__
    return input_name, input_shape.tolist(), input_dtype


def convert_to_relay(tflite_model, input_name: str, input_shape, input_dtype):
    shape_dict = {input_name: input_shape}
    dtype_dict = {input_name: input_dtype}
    mod, params = from_tflite(tflite_model, shape_dict=shape_dict, dtype_dict=dtype_dict)
    return mod, params


def compile_model_to_c(mod, params, output_dir="cpp_inference/tvm_model_c", mcpu="cortex-a65"):
    target = tvm.target.Target(f"c -keys=arm_cpu,cpu -mcpu={mcpu}")

    executor = Executor("aot", {
        "unpacked-api": True,
        "interface-api": "c",
        "workspace-byte-alignment": 8
    })
    runtime = Runtime("crt")

    with tvm.transform.PassContext(opt_level=3, config={
        "tir.disable_vectorize": True
    }):
        lib = relay.build(mod, target=target, params=params, executor=executor, runtime=runtime)

    export_model_library_format(lib, f"{output_dir}.tar")
    print(f"[INFO] Exported model to {output_dir}.tar")


def main():
    model_path = "runs/trained_models/train_0613.tflite"

    tflite_model = load_tflite_model(model_path)
    input_name, input_shape, input_dtype = get_input_details(model_path)

    mod, params = convert_to_relay(tflite_model, input_name, input_shape, input_dtype)

    print("Relay IR module:")
    print(mod)

    compile_model_to_c(mod, params, output_dir="cpp_inference/tvm_model_c")


if __name__ == "__main__":
    main()
