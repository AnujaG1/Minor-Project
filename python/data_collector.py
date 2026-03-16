# data_collector.py
# PURPOSE: Saves all received data to CSV file
#          This CSV is used later for RL training
#          Run this ONCE with full simulation to collect training data

import csv
import os
import time
import sys

# Always use absolute paths regardless of where script is run from
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# ↑ /home/anuja/MinorProject/workspace/RL/python

RL_DIR = os.path.dirname(SCRIPT_DIR)
# ↑ /home/anuja/MinorProject/workspace/RL

RESULTS_DIR = os.path.join(RL_DIR, "results")
# ↑ /home/anuja/MinorProject/workspace/RL/results

sys.path.append(RL_DIR)
# ↑ So imports work from anywhere

from python.udp_receiver import UDPReceiver
from python.feature_extractor import FeatureExtractor

def collect_training_data(
        output_file="results/training_data.csv",
        duration=110  
):
    os.makedirs("results", exist_ok=True)

    #create results directory
    os.makedirs("results", exist_ok=True)

    # setup receiver and extractor
    receiver = UDPReceiver(host="127.0.0.1", port=9999)
    extractor = FeatureExtractor()


    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # write header
        writer.writerow([
            'sim_time',
            'node',
            'node_type',
            'pkt_rate',
            'pkt_size',
            'interval',
            'is_attacker',
            'port',
            'f1_pkt_rate',
            'f2_pkt_size',
            'f3_interval',
            'f4_port',
            'f5_time',
            'f6_delta'
        ])

        print(f"Saving training data to {output_file}")
        print("Start OMNeT++ simulation now!")
        print("-" * 50)

        #start receiver
        receiver.start()

        start_time = time.time()
        rows_written = 0

        while time.time() - start_time < duration:
            data_batch = receiver.get_data()            # get all queued UDP packets

            for raw in data_batch:
                state, label, node = extractor.extract(raw)    # Get all queued data

                # write to CSV
                writer.writerow([
                    raw.get('time', 0),
                    raw.get('node', ''),
                    raw.get('type', ''),
                    raw.get('pkt_rate', 0),
                    raw.get('pkt_size', 0),
                    raw.get('interval', 0),
                    label,
                    raw.get('port', 0),
                    state[0],  # f1
                    state[1],  # f2
                    state[2],  # f3
                    state[3],  # f4
                    state[4],  # f5
                    state[5]   # f6
                ])
                rows_written += 1

                #print progress
                if raw.get('pkt_rate', 0) > 100:
                    print(f"t={raw['time']:.2f}s | "
                          f"{raw['node']:<15} | "
                          f"ATTACK! rate={raw['pkt_rate']:.0f}")
                    
            csvfile.flush()
            # ↑ Write to disk regularly

            time.sleep(0.01)
            # ↑ Check every 10ms

            # Check if simulation ended
            if receiver.get_stats()["sim_ended"]:
                print("Simulation ended - stopping collection")
                break

        receiver.stop()
    print(f"\nData collection complete!")
    print(f"Rows saved: {rows_written}")
    print(f"File: {output_file}")
    return output_file

if __name__ == "__main__":
    collect_training_data()
        
