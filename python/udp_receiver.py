import socket
import json
import threading
import queue
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'

)
logger = logging.getLogger(__name__)
class UDPReceiver:
    def __init__(self, host="127.0.0.1", port=9999, buffer_size=10000):
        self.host = host
        self.port = port
        self.buffer_size = buffer_size

        self.data_queue = queue.Queue(maxsize=buffer_size)
        self.running = False
        self.thread = None
        self.stats = {
            "packets_received": 0,
            "packets_dropped": 0,
            "parse_errors": 0,
            "sim_ended": False
        }

    def start(self):
        self.running = True
        self.thread = threading.Thread(
            target=self._listen_loop,
            daemon=True
        )
        self.thread.start()

        logger.info(f"UDP Receiver started on {self.host}:{self.port}")

    def stop(self):
        self.running = False
        logger.info("UDP Receiver stopped")

    def _listen_loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.host, self.port))
        sock.settimeout(1.0)

        logger.info(f"Socket bound to {self.host}:{self.port}")
        logger.info("Waiting for OMNeT++ simulation data...")

        while self.running:
            try:
                raw_data, addr = sock.recvfrom(4096)
                json_str = raw_data.decode('utf-8')

                if '"type":"SIM_END"' in json_str:
                    logger.info("Simulation ended signal received")
                    self.stats["sim_ended"] = True
                    continue
                parsed = json.loads(json_str)

                try:
                    self.data_queue.put_nowait(parsed)
                    self.stats["packets_received"] += 1

                except queue.Full:
                    self.data_queue.get_nowait()
                    self.data_queue.put_nowait(parsed)
                    self.stats["packets_dropped"] +=1

            except socket.timeout:
                continue

            except json.JSONDecodeError as e:
                self.stats["parse_errors"] += 1
                logger.warning(f"Bad JSON received: {e}")
                continue
            except Exception as e:
                logger.error(f"Receiver error: {e}")
                break

        sock.close()

    def get_data(self):
        items = []
        while not self.data_queue.empty():
            try:
                items.append(self.data_queue.get_nowait())

            except queue.Empty:
                break                  # Queue is empty - stop collecting
        
        return items
    
    def get_stats(self):
        return self.stats.copy()
     # ↑ .copy() returns new dict - prevents external modification
