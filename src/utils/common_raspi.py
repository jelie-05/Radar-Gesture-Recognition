import numpy as np
from cv2 import resize, INTER_AREA


def do_preprocessing(range_doppler):
    # Normalizing to [0, 1]

    range_doppler = np.abs(range_doppler)
    for index, channel in enumerate(range_doppler):
        min = np.min(channel)
        max = np.max(channel)
        range_doppler[index] = (channel - min) / (max - min)

    range_doppler = np.transpose(range_doppler, (2, 1, 0))
    range_doppler = resize(range_doppler, dsize=(32, 32), interpolation=INTER_AREA)
    range_doppler = np.transpose(range_doppler, (2, 1, 0))

    return range_doppler

def do_inference_processing_np(range_doppler: np.array):
    range_doppler = do_preprocessing(range_doppler)

    range_doppler = range_doppler.astype(np.float32)
    range_doppler = np.expand_dims(range_doppler, axis=0)  # Add batch dimension

    return range_doppler