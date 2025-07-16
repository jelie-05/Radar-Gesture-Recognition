import os
import tarfile
import numpy as np
import tvm
from tvm import relay
from tvm.relay.frontend import from_tflite
from tvm.runtime import save_param_dict
from tvm.contrib import graph_executor
from tflite import Model
import tflite_runtime.interpreter as tflite
from torch.utils.data import Subset
from tvm.micro.testing.utils import create_header_file

from src.train_utils.dataset import RadarGestureDataset, DataGenerator


def load_tflite_model(model_path: str):
    with open(model_path, "rb") as f:
        model_buf = f.read()
    return Model.GetRootAsModel(model_buf, 0)


def get_input_details(model_path: str):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    return input_details[0]["name"]


def convert_to_relay(tflite_model, input_name, input_shape, input_dtype):
    shape_dict = {input_name: input_shape}
    dtype_dict = {input_name: input_dtype}
    mod, params = from_tflite(tflite_model, shape_dict=shape_dict, dtype_dict=dtype_dict)
    return mod, params


def compile_model(mod, params, target="llvm"):
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)
    return lib


def find_one_sample_per_class(dataset, num_classes):
    """Returns a dict of class_id -> dataset index"""
    class_to_index = {}
    for i in range(len(dataset)):
        # print(f"dataset i: {dataset[i]}")
        label = dataset[i][-1]

        label = int(label)
        if label not in class_to_index:
            class_to_index[label] = i
        if len(class_to_index) == num_classes:
            break
    return class_to_index


def main():
    model_path = "runs/trained_models/train_0613.tflite"
    input_shape = [1, 2, 32, 10]
    input_dtype = "float32"
    num_classes = 4  

    # === TVM Setup ===
    tflite_model = load_tflite_model(model_path)
    input_name = get_input_details(model_path)
    mod, params = convert_to_relay(tflite_model, input_name, input_shape, input_dtype)
    lib = compile_model(mod, params)
    dev = tvm.cpu()
    module = graph_executor.GraphModule(lib["default"](dev))

    # === Dataset Setup ===
    dataset = RadarGestureDataset(root_dir="data/recording", annotation_csv="annotation")
    class_to_index = find_one_sample_per_class(dataset, num_classes)

    # === Saving Directory ===
    save_dir = "data/samples"
    os.makedirs(save_dir, exist_ok=True)
    tar_path = os.path.join(save_dir, "test_samples.tar")

    with tarfile.open(tar_path, mode="w") as tar_file:
        for class_id, idx in class_to_index.items():
            # Preprocess using DataGenerator
            subset = Subset(dataset, [idx])
            loader = DataGenerator(
                subset,
                batch_size=1,
                shuffle=False,
                max_length=input_shape[-1],
                num_workers=0,
                drop_last=False
            ).get_loader()

            batch = next(iter(loader))
            input_tensor = batch["rdtm"].numpy()  # shape: (1, 2, 32, 10)

            # Run inference
            module.set_input(input_name, input_tensor)
            module.run()
            logits = module.get_output(0).numpy()
            softmax = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

            # Remove batch dimension for C export
            input_tensor = input_tensor[0]
            logits = logits[0]
            softmax = softmax[0]

            class_name = f"class{class_id}"
            create_header_file(f"{class_name}_input", input_tensor, save_dir, tar_file)
            create_header_file(f"{class_name}_logits", logits, save_dir, tar_file)
            create_header_file(f"{class_name}_softmax", softmax, save_dir, tar_file)

    print(f"Saved 1 sample per class to: {tar_path}")


if __name__ == "__main__":
    main()
