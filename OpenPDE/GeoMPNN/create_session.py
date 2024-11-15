from lips.benchmark.airfransBenchmark import AirfRANSBenchmark
from lips.dataset.airfransDataSet import AirfRANSDataSet, extract_dataset_by_simulations
import dill
import os
import numpy as np

import json

ROOT = "."

DIRECTORY_NAME = os.path.join(ROOT, 'Dataset')
BENCHMARK_NAME = "Case1"
LOG_PATH = os.path.join(ROOT, "logs", "lips_logs.log")
BENCH_CONFIG_PATH = os.path.join(ROOT, "airfoilConfigurations", "benchmarks","confAirfoil.ini") #Configuration file related to the benchmark

print(f"DIRECTORY_NAME: {DIRECTORY_NAME}")
print(f"BENCHMARK_NAME: {BENCHMARK_NAME}")
print(f"LOG_PATH: {LOG_PATH}")
print(f"BENCH_CONFIG_PATH: {BENCH_CONFIG_PATH}")

if not os.path.exists(os.path.dirname(LOG_PATH)):
    os.makedirs(os.path.dirname(LOG_PATH))

print("Preparing benchmark")
benchmark=AirfRANSBenchmark(benchmark_path = DIRECTORY_NAME,
                            config_path = BENCH_CONFIG_PATH,
                            benchmark_name = BENCHMARK_NAME,
                            log_path = LOG_PATH)
benchmark.load(path=DIRECTORY_NAME)

print("Saving")
dill.dump_session(os.path.join(ROOT, f"benchmark_{BENCHMARK_NAME}_session"))