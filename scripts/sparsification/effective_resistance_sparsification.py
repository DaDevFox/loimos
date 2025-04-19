#!/usr/bin/env python3
import io
from os import system
import subprocess
import os
import numpy as np
import pandas as pd
from EffectiveResistanceSampling.Network import Network
import networkx as nx
import argparse
from time import perf_counter
from multiprocessing import Pool, set_start_method
from itertools import starmap

import multiprocessing
import platform
import os
import cupy
from cupy import cuda
from mpi4py import MPI


# reference scripts:
# location_herustics.py
#

# CuPy + CUDA Kernels:
# CuPy is a flexible library that provides NumPy-like functionality on GPUs using CUDA. You can write custom CUDA kernels for optimal performance.

# cuBLAS/cuSPARSE/cuSOLVER from CUDA Toolkit:
# NVIDIA’s libraries (cuBLAS, cuSPARSE, cuSOLVER) offer high-performance linear algebra operations, including iterative solvers.

# scikit-cuda or PyCUDA:
# These wrappers around CUDA libraries let you handle allocations and computations manually.

# PETSc or MAGMA for Large-Scale CG:
# These specialized libraries offer high-performance solvers for scientific computing.

def worker_function(gpu_id, rank, hostname, *args):
    """Function executed by each worker process."""
    print(f"Rank {rank} on {hostname} using GPU {gpu_id}")
    # with cuda.Device(gpu_id):
    #     cupy.asarray(12345)  # Simple GPU operation
    process_subset(gpu_id, *args)

def parse_args():
    parser = argparse.ArgumentParser()

    # Positional/required arguments:
    parser.add_argument(
        "input_dir",
        metavar="I",
        help="The path to directory containing data files for a population",
    )
    parser.add_argument(
        "output_dir",
        metavar="O",
        help="The directory in which the output files should be saved",
    )

    parser.add_argument(
        "resultant_sample_size",
        metavar="Q",
        type=float,
        help="approximate percentage of number of edges to maintain in the sparsified network",
    )

    parser.add_argument(
        "-s", "--split",
        type=int,
        default=1,
        help="Run sparsification on S separate equally-sized subintervals",
    )

    parser.add_argument(
        "-p", "--parallelize",
        action="store_true",
        help="If -s or --split is set to an integer value, run the [-s/--split value] subinterval calculations in parallel rather than in serial sequence",
    )

    parser.add_argument(
        "--process_count",
        default=0,
        type=int,
        help="Number of processes to use for parallelization. Default is the number of subintervals specified by -s/--split. ",
    )

    parser.add_argument(
        "--time",
        default="times.csv",
        type=str,
        help="time and store the timing results in this csv file",
    )

    parser.add_argument(
        "-t", "--test-mode",
        action="store_true",
        help="run in experimental/debug mode in which only first 10000 lines of visits.csv are read",
    )

    args = parser.parse_args()

    if args.parallelize and args.process_count == 0:
        args.process_count = args.split


    return args

def process_subset(gpu_id, df_subset, q, epsilon=0.1, method='kts'):
    with cuda.Device(gpu_id):
        edge_list = df_subset[['pid', 'lid']].dropna().astype(int).to_numpy()  # should be 2 x m shape
        weights = df_subset['duration'].dropna().to_numpy()  # weight edge by visit duration

        # Time the Network constructor
        print("running network constructor")
        start = perf_counter()
        network = Network(edge_list, weights)
        network_constructor_time = perf_counter() - start
        print("network constructor complete")

        # Time the effective resistance calculation
        print("running effective resistance", flush=True)
        start = perf_counter()
        Effective_R = network.effR(epsilon, method)
        effR_time = perf_counter() - start
        print("effective resistance complete", flush=True)

        # Time the sparsification process
        print("running network.spl", flush=True)
        start = perf_counter()
        print(f"q: {q}, Effective_R: {Effective_R}, seed: 2020", flush=True)
        EffR_Sparse = network.spl(q, Effective_R, seed=2020)
        spl_time = perf_counter() - start
        print("network.spl complete", flush=True)

        print(f"marginal times: {network_constructor_time}, {effR_time}, {spl_time}", flush=True)

        sparse_edge_set = set(map(tuple, EffR_Sparse.E_list.tolist()))
        filtered_df_subset = df_subset[
            df_subset[["pid", "lid"]].apply(tuple, axis=1).isin(sparse_edge_set)
        ]


    return filtered_df_subset, network_constructor_time, effR_time, spl_time

def main():
    global args
    args = parse_args()

    script_start = perf_counter()

    preprocessing_time = 0
    network_constructor_time = 0
    effR_time = 0
    spl_time = 0

    if not os.path.exists(args.input_dir):
        print(f'input directory not found: {args.input_dir}', flush=True)
        raise FileNotFoundError(args.input_dir)

    input = os.path.join(args.input_dir, 'visits.csv')
    if args.test_mode:
        df = pd.read_csv(input, nrows=1000)
    else:
        df = pd.read_csv(input)


    if args.split is not None:
        hostname = platform.node()

        COMM = MPI.COMM_WORLD
        RANK = COMM.Get_rank()
        SIZE = COMM.Get_size()

        num_subsets = SIZE if args.parallelize else args.split

        times = {}

        days = df.groupby('daynum')
        num_days = len(days.groups.keys())
        seconds_in_day = (24 * 60.0 * 60.0)

        subset_size = (seconds_in_day * num_days) / num_subsets
        print("Sparsifying:", flush=True)
        print(f"{num_subsets} intervals of length {subset_size} seconds each", flush=True)
        def subset_num(col):
            return np.floor(col / subset_size)

        if args.parallelize:
            df.where(subset_num(df['start_time']) == RANK, inplace=True)

        # temp
        start = perf_counter()
        df['duration'] = df['duration'].mask(
                            subset_num(df['start_time']) != subset_num(df['end_time']),
                            subset_size * (subset_num(df['start_time']) + 1) - df['start_time'])
        df['end_time'] = df['end_time'].mask(
                            subset_num(df['start_time']) != subset_num(df['end_time']),
                            df['start_time'] + df['duration'])
        preprocessing_time = perf_counter() - start
        # TODO: spawn split-off days into other subset dataframes

        GPUS_PER_NODE = cupy.cuda.runtime.getDeviceCount()
        GPUS_PER_NODE = min(GPUS_PER_NODE, SIZE)

        if args.parallelize:
            print(f"PARALLELIZED job {RANK}/{SIZE} (machine with {cupy.cuda.runtime.getDeviceCount()} GPUs) on gpuID={RANK % GPUS_PER_NODE}", flush=True)

        # filtered_dfs = []
        # results = [None] * len(subsets)

        start = perf_counter()
        result = None

        if args.parallelize:
            # spawn_workers_per_node(lambda gpu_id: (subset_args[gpu_id]))
            result = process_subset(RANK % GPUS_PER_NODE, df, int(args.resultant_sample_size * len(df)))
            print("test0")
        else:
            subsets = df.groupby(subset_num(df['start_time']))
            subset_args = [[0, subset, int(args.resultant_sample_size * len(subset))] for _, subset in subsets]
            results = list(starmap(process_subset, subset_args))

        if not args.parallelize:
            # Combine results and update global times
            filtered_dfs = []
            for result in results:
                filtered_dfs.append(result[0])
                network_constructor_time += result[1]
                effR_time += result[2]
                spl_time += result[3]
            print(f'Spent {perf_counter() - start}s sparsifying data')
            final_filtered_df = pd.concat(filtered_dfs)
    else:
        q = int(args.resultant_sample_size * float(len(df)))
        final_filtered_df = process_subset(0, df, q, None, None)

    print("Test")
    # Gather results from all ranks to rank 0
    if args.parallelize:
        # Each process prepares its filtered_df and timing info
        local_filtered_df = result[0]
        local_times = {
                    "preprocessing_time": preprocessing_time,
                    "network_constructor_time": result[1],
                    "effR_time": result[2],
                    "spl_time": result[3],
                    "total": perf_counter() - script_start,
                                }

        print("Test 2")
        # Serialize DataFrame to bytes for MPI
        local_bytes = local_filtered_df.to_csv(index=False).encode()
        local_size = np.array([len(local_bytes)], dtype=np.int32)
        print("sending data")
        sizes = COMM.allgather(local_size)
        print("sent")
        # Gather all DataFrame byte sizes
        max_size = max(s[0] for s in sizes)
        # Pad bytes to max_size for MPI
        padded = np.zeros(max_size, dtype=np.uint8)
        padded[: local_size[0]] = np.frombuffer(local_bytes, dtype=np.uint8)
        gathered = None
        if RANK == 0:
            gathered = np.empty((SIZE, max_size), dtype=np.uint8)
            print("gathered")
        COMM.Gather(padded, gathered, root=0)
        # Gather timing info
        all_times = COMM.gather(local_times, root=0)
        if RANK == 0:
            # Reconstruct DataFrames
            dfs = []
            for i in range(SIZE):
                sz = sizes[i][0]
                df_bytes = bytes(gathered[i][:sz])
                dfs.append(pd.read_csv(io.StringIO(df_bytes.decode())))
            final_filtered_df = pd.concat(dfs, ignore_index=True)
            # Write mega visits.csv
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            final_filtered_df.to_csv(os.path.join(args.output_dir, "visits.csv"), index=False)
            # Write times.csv with one row per process
            times_df = pd.DataFrame(all_times)
            times_df.to_csv(args.time, index=False)
            print(
                f"complete: {os.path.join(args.output_dir, 'visits.csv')}",
                flush=True,
            )
        else:
# Write times.csv as before (single row)
            times = {
                "preprocessing_time": preprocessing_time,
                "network_constructor_time": network_constructor_time / num_subsets,
                "effR_time": effR_time / num_subsets,
                "spl_time": spl_time / num_subsets,
                "total": perf_counter() - script_start,
                            }
            times_df = pd.DataFrame([times])
            times_df.to_csv(args.time, index=False)
            print(
            f"complete: {os.path.join(args.output_dir, 'visits.csv')}", flush=True
            )
                                                                                                                                                                        

if __name__ == "__main__":
    main()
