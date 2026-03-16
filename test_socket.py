import socket, json, signal, sys

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("127.0.0.1", 9999))
sock.settimeout(10.0)  # longer timeout

print("Listening on port 9999...")
print("-" * 60)

def handle_exit(sig, frame):
    print("\nStopped.")
    sock.close()
    sys.exit(0)

signal.signal(signal.SIGINT, handle_exit)

while True:  # ← run forever until Ctrl+C or SIM_END
    try:
        data, _ = sock.recvfrom(4096)
        msg = json.loads(data.decode())

        # Check for simulation end signal
        if msg.get("type") == "SIM_END":
            print("=== Simulation ended ===")
            break

        # Show ALL data (remove the pkt_rate > 0 filter for now)
        print(f"t={msg['time']:.3f}s | "
              f"{msg['node']:<15} | "
              f"rate={msg['pkt_rate']:>8.1f} pkt/s | "
              f"attacker={msg['is_attacker']}")

    except socket.timeout:
        print("Waiting...")
    except json.JSONDecodeError:
        pass  # ignore malformed packets

sock.close()