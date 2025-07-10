import socket
import numpy as np
import time

HOST = '127.0.0.1'
PORT = 5005

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.connect((HOST, PORT))
    while True:  # send 100 frames (change as needed, or use while True for infinite)
        frame = np.random.rand(1, 2, 32, 10).astype(np.float32)
        data_bytes = frame.tobytes()
        frame_size = len(data_bytes)
        sock.sendall(frame_size.to_bytes(4, 'big'))  # send 4-byte length
        sock.sendall(data_bytes)
        print(f"size: {frame_size} bytes")
        time.sleep(0.5)  # send a frame every 0.5 second (adjust as you like)

    