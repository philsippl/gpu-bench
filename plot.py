import os
import json
import re
import matplotlib.pyplot as plt

def find_benchmark_dirs(root_dir):
    """
    Recursively search for benchmark directories and return a list of their paths.
    """
    benchmark_dirs = []
    for subdir, dirs, files in os.walk(root_dir):
        for dir in dirs:
            if "→" in dir:
                benchmark_dirs.append(os.path.join(subdir, dir))
    return benchmark_dirs

def extract_batch_size(dir_name):
    """
    Extract the batch size from the directory name.
    """
    match = re.search(r'\(\d+\s+x\s+(\d+)\)', dir_name)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Could not extract batch size from directory name: {dir_name}")

def load_benchmarks(benchmark_dirs):
    """
    Load benchmark data from the list of benchmark directory paths.
    """
    benchmarks = {}
    for bench_dir in benchmark_dirs:
        batch_size = extract_batch_size(bench_dir)
        estimates_file = os.path.join(bench_dir, 'new', 'estimates.json')
        with open(estimates_file, 'r') as f:
            data = json.load(f)
        mean_estimate = data['mean']['point_estimate']
        estimates_file = os.path.join(bench_dir, 'new', 'benchmark.json')
        with open(estimates_file, 'r') as f:
            data = json.load(f)
        elements = data['throughput']['Elements']
        bench_name = bench_dir.split("/")[2]
        if bench_name not in benchmarks:
            benchmarks[bench_name] = {"x": [], "y": []}
        benchmarks[bench_name]["y"].append(1/(mean_estimate/elements/1e9)/1e6)
        benchmarks[bench_name]["x"].append(batch_size/31)
        print(1/(mean_estimate/elements/1e9))
    return benchmarks

def plot_benchmarks(benchmarks, output_file):
    plt.figure(figsize=(10, 6))
    for bench, values in benchmarks.items():
        print(bench)
        plt.plot(values["x"], values["y"], label=bench, marker='.')

    plt.xlabel('Batch Size')
    plt.ylabel('Million Queries/s')
    plt.title('Matmul')
    plt.legend()
    plt.grid(True)
    plt.ylim(ymin=0)
    plt.savefig(output_file, bbox_inches='tight')

# Usage
root_dir = 'target/criterion'
benchmark_dirs = find_benchmark_dirs(root_dir)
print(benchmark_dirs)
benchmarks = load_benchmarks(benchmark_dirs)
output_file = 'xxx.png'
plot_benchmarks(benchmarks, output_file)

print(f"Benchmark plot saved to {output_file}")
