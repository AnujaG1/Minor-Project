# run.py  — create this file in your python/ folder
from realtime_pipeline import RealtimePipeline
import time

pipeline = RealtimePipeline(model_path="results/ddqn_model.pth")
pipeline.start()

print("Pipeline running — start your OMNeT++ simulation now!")
print("Press Ctrl+C to stop.\n")

try:
    while True:
        time.sleep(5)
        snap = pipeline.state.snapshot()
        print(f"Counters: {snap['counters']}")
except KeyboardInterrupt:
    pipeline.stop()
    print("Stopped.")