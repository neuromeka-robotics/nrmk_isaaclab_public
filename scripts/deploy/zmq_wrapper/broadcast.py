import zmq
"""
broadcast.py
A simple ZeroMQ-based PUB/SUB broadcast server and client for sending and receiving Python (including NumPy) objects over the network using msgpack serialization.
Classes:
    BroadcastServer:
        A publisher (server) that broadcasts serialized Python objects to all connected subscribers using ZeroMQ PUB sockets.
        Args:
            ip (str): IP address to bind the server to. Defaults to "localhost".
            port (int): Port to bind the server to. Defaults to 5557.
            log_level (int): Logging level. Defaults to logging.DEBUG.
        Methods:
            broadcast(data: bytes):
                Serializes and sends data to all subscribers.
            close():
                Closes the ZeroMQ socket and terminates the context.
    BroadcastClient:
        A subscriber (client) that connects to a broadcast server and receives serialized Python objects using ZeroMQ SUB sockets.
        Args:
            ip (str): IP address of the broadcast server.
            port (int): Port of the broadcast server. Defaults to 5557.
            log_level (int): Logging level. Defaults to logging.DEBUG.
            compression (str, optional): Compression method (currently unused).
        Methods:
            async_start(callback: Callable[[dict], None], poll_timeout_ms: int = 5):
                Starts an asynchronous listener thread that calls the provided callback with each received message.
            stop():
                Stops the listener thread and closes the ZeroMQ socket and context.
        # BroadcastClient is a ZeroMQ SUB client that connects to a broadcast server,
        # receives serialized messages (including NumPy arrays), and processes them
        # asynchronously using a user-provided callback function.
Usage:
    Run as a server:
        python broadcast.py --server --port 6000
    Run as a client:
        python broadcast.py --client --ip localhost --port 6000
"""


#pip install pyzmq

from typing import Tuple, Callable
import pickle
# import zlib
# import lz4.frame
import argparse
import logging
import threading
import numpy as np

import msgpack
import msgpack_numpy as m

# When client is using numpy 1.x and server is using numpy 2.x, client can't pickle the data
# This helps NumPy 1.x unpickle NumPy 2.x pickles, please use this sparingly and only temporarily
# From https://github.com/numpy/numpy/issues/28340
if np.__version__[:2] == "1.":
    import sys
    sys.modules["numpy._core.numeric"] = np.core.numeric
    sys.modules["numpy._core.multiarray"] = np.core.multiarray

# def make_compression_method(compression: str) -> Tuple[Callable, Callable]:
#     """
#     NOTE: lz4 is faster than zlib, but zlib has better compression ratio
#         :return: compress, decompress functions
#             def compress(object) -> bytes
#             def decompress(data) -> object
    # """
        
    # if compression == 'lz4':
    #     def compress(data): return lz4.frame.compress(pickle.dumps(data))
    #     def decompress(data): return pickle.loads(lz4.frame.decompress(data))
    # elif compression == 'zlib':
    #     def compress(data): return zlib.compress(pickle.dumps(data))
    #     def decompress(data): return pickle.loads(zlib.decompress(data))
    # else:
    #     raise Exception(f"Unknown compression algorithm: {compression}")
    # return compress, decompress


class ZmqPublisher:
    def __init__(self,
                 ip="localhost",
                 port=5557,
                 log_level=logging.DEBUG,
                #  compression: str = 'lz4'
                 ):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 3)  # queue size 3 for send buffer
        self.socket.setsockopt(zmq.LINGER, 0)  # Clean pending messages in the socket buffer
        self.socket.bind(f"tcp://{ip}:{port}")
        logging.basicConfig(level=log_level)
        self.port = port
        logging.info(f"[ZmqPublisher] Broadcasting on tcp://{ip}:{port}")
        
    def _serialize(self, data) -> bytes:
        """Serialize data to bytes using msgpack."""

        try:
            return msgpack.packb(data, use_bin_type=True, default=m.encode)
        except (TypeError, msgpack.PackException) as e:
            raise TypeError(f"Data of type {type(data)} is not serializable: {e}")
        
    def broadcast(self, data: bytes):
        """Send a raw message (already serialized) over ZMQ."""
        message = self._serialize(data)
        self.socket.send(message)
        
    def close(self):
        self.socket.close()
        self.context.term()
        logging.info("[ZmqPublisher] Socket closed")
        
        

class ZmqSubscriber:
    def __init__(self,
                 ip: str, port=5557,
                 log_level=logging.DEBUG, 
                 compression: str = None,
                 verbose: bool = False):
        
        self.verbose = verbose
        self.address = f"tcp://{ip}:{port}"
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(self.address)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all messages
        self.socket.setsockopt(zmq.RCVHWM, 3)  # queue size 3 for receive buffer
        self.socket.setsockopt(zmq.CONFLATE, 1)
        
        # # _, self.decompress = make_compression_method(comppression)
        # # Set a timeout for the recv method (e.g., 1.5 second)
        # self.socket.setsockopt(zmq.RCVTIMEO, 1500)


        self.stop_event = threading.Event()
        self.thread = None

        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)
        
                                     
        logging.basicConfig(level=log_level)
        logging.debug(f"ZmqSubscriber client is connecting to tcp://{ip}:{port}")
        
    def __del__(self):
        self.stop()

    def _deserialize(self, msg: bytes):
        """Deserialize msgpack-encoded data, including numpy arrays."""
        try:
            return msgpack.unpackb(msg, raw=False, object_hook=m.decode)
        except Exception as e:
            raise ValueError(f"Failed to deserialize message: {e}")
  
    def async_start(self, callback: Callable[[dict], None], poll_timeout_ms: int = 5):
        def async_listen():
            timeout_count = 0
            while not self.stop_event.is_set():
                socks = dict(self.poller.poll(poll_timeout_ms))
                if self.socket in socks and socks[self.socket] == zmq.POLLIN:
                    try:
                        msg = self.socket.recv()
                        data = self._deserialize(msg)
                        callback(data)
                        timeout_count = 0
                    except Exception as e:
                        logging.exception(f"[ZmqSubscriber] Error in recv: {e}")
                else:
                    timeout_count += 1
                    if self.verbose and timeout_count % 50 == 0:
                        logging.warning(f"[ZmqSubscriber] No data for {timeout_count * poll_timeout_ms} ms on {self.address}")

        self.thread = threading.Thread(target=async_listen, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)  # timeout avoids infinite blocking
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()
        logging.info(f"[BroadcastClient] Cleanly stopped subscriber for {self.address}")

##############################################################################

if __name__ == "__main__":
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument('--server', action='store_true')
    parser.add_argument('--client', action='store_true')
    parser.add_argument('--ip', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=6000)
    args = parser.parse_args()

    if args.server:
        ps = ZmqPublisher(port=args.port)
        while True:
            ps.broadcast({'message': "Hello World"})
            time.sleep(1)
    elif args.client:
        pc = ZmqSubscriber(ip=args.ip, port=args.port)
        pc.async_start(callback=lambda x: print(x))
        print("Listening... asynchonously")
    else:
        raise Exception('Must specify --server or --client')
