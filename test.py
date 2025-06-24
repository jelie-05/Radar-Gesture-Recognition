import tarfile

tar_path = "data/samples/test_samples.tar"

with tarfile.open(tar_path, "r") as tar:
    tar.list()  # List contents