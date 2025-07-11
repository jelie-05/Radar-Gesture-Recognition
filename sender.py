# import socket
# import numpy as np
# import time

# HOST = '127.0.0.1'
# PORT = 5005

# with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
#     sock.connect((HOST, PORT))
#     while True:  # send 100 frames (change as needed, or use while True for infinite)
#         frame = np.random.rand(1, 2, 32, 10).astype(np.float32)
#         data_bytes = frame.tobytes()
#         frame_size = len(data_bytes)
#         sock.sendall(frame_size.to_bytes(4, 'big'))  # send 4-byte length
#         sock.sendall(data_bytes)
#         print(f"size: {frame_size} bytes")
#         time.sleep(0.5)  # send a frame every 0.5 second (adjust as you like)


import socket
import struct

HOST = '0.0.0.0'
PORT = 5006


def recv_all(sock, size):
    data = b''
    while len(data) < size:
        more = sock.recv(size - len(data))
        if not more:
            raise EOFError("Connection closed")
        data += more
    return data

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
    server.bind((HOST, PORT))
    server.listen(1)
    print("Waiting for connection...")
    conn, addr = server.accept()
    with conn:
        print(f"Connected by {addr}")

        # Option 1: If you sent just the 4 floats (no length)
        while True:
            data = recv_all(conn, 4*4)  # 4 floats, each 4 bytes
            floats = struct.unpack('!4f', data)  # Network byte order
            print(f"Received floats: {floats}")

        # Option 2: If you sent a length header first
        # data_len = struct.unpack('!I', recv_all(conn, 4))[0]
        # data = recv_all(conn, data_len)
        # floats = struct.unpack(f'!{data_len // 4}f', data)
        # print(f"Received floats: {floats}")
