
class DebouncerTime:
    def __init__(self, detect_threshold=0.6, noise_threshold=0.3, memory_length=30, min_num_detections=3):

        self.detect_threshold = detect_threshold
        self.noise_threshold = noise_threshold
        self.memory_length = memory_length
        self.min_num_detections = min_num_detections

        self.dtm_memory = []
        self.rtm_memory = []
        self.detection_memory = []

    def add_scan(self, frame):
        if len(self.dtm_memory) >= self.memory_length:
            self.dtm_memory.pop(0)
            self.rtm_memory.pop(0)

        # Only add the first channel !!!
        processed_frame = frame[0, 0, :, :]
        max_value = processed_frame.max()

        h, w = (processed_frame == max_value).nonzero(as_tuple=True)
        h, w = h[0], w[0]

        rtm = processed_frame[:, w].unsqueeze(1)  # Range-Time Map
        dtm = processed_frame[h, :].unsqueeze(1)

        self.dtm_memory.append(dtm)
        self.rtm_memory.append(rtm)

    def get_scans(self):
        return self.dtm_memory, self.rtm_memory