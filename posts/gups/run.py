# SPDX-FileCopyrightText: Copyright (c) 2020-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

import subprocess
import csv

test_type_to_id = {
    "updates": 0, 
    "reads": 1,
    "writes": 2,
    "reads_writes": 3,
    "updates_no_loop": 4
}

DEFAULT_GLOBAL_GUPS_SIZE=29

def run_test(input_size, device_id, test_type, occupancy, repeats, memory_loc):
    tests_to_run = [] 
    if test_type == "all":
        tests_to_run = [0, 1, 2, 3, 4]
    else:
        tests_to_run.append(test_type_to_id[test_type])

    results = []

    if memory_loc == "global":
        for t in tests_to_run:
            proc = subprocess.run(
                [
                    "./gups",
                    "-n", str(input_size),
                    "-t", str(t),
                    "-d", str(device_id),
                    "-o", str(occupancy),
                    "-r", str(repeats)
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )

            if proc.returncode != 0:
                print(proc.stdout)
                raise RuntimeError("Failed to run GUPS")

            output = proc.stdout
            for output_line in output.splitlines():
                if "Result" in output_line:
                    output_line_split = output_line.split()
                    assert output_line_split[0] == "Result"

                    value = output_line_split[-1]
                    res_unit = output_line_split[1]
                    op_name = output_line_split[2]
                    results.append([res_unit, op_name, value])        
    else: 
        for t in tests_to_run:
            proc = subprocess.run(
                [
                    "./gups",
                    "-s", str(input_size),
                    "-t", str(t),
                    "-d", str(device_id),
                    "-r", str(repeats)
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )

            if proc.returncode != 0:
                print(proc.stdout)
                raise RuntimeError("Failed to run GUPS")

            output = proc.stdout
            for output_line in output.splitlines():
                if "Result" in output_line:
                    output_line_split = output_line.split()
                    assert output_line_split[0] == "Result"

                    value = output_line_split[-1]
                    res_unit = output_line_split[1]
                    op_name = output_line_split[2]
                    results.append([res_unit, op_name, value])        

    with open('results.csv', "a") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        row = [2**input_size]
        for result in results:
            row += [result[2]]
        csv_writer.writerow(row)

def main():
    # Parse command line arguments
    args = parse_commandline_argument()
    device_id = int(args.device_id)
    test_type = args.test
    occupancy = args.occupancy
    repeats = args.repeats
    input_size_begin = int(args.input_size_begin)
    input_size_end = int(args.input_size_end)
    memory_loc = args.memory_loc

    if memory_loc == "global" and input_size_begin == 0:
        input_size_begin = DEFAULT_GLOBAL_GUPS_SIZE
    if memory_loc == "global" and input_size_end == 0:
        input_size_end = DEFAULT_GLOBAL_GUPS_SIZE

    # Write header
    with open('results.csv', "w") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        header = ["Size"]
        if test_type == "all":
            if memory_loc == "global":
                for t in test_type_to_id:
                    header += [t]
            else:
                for t in test_type_to_id:
                    header += ['shmem_GPU_'+t]
                    header += ['shmem_SM_'+t]
        else:
            if memory_loc == "global":
                header += [test_type]
            else:
                header += ['shmem_GPU_'+test_type]
                header += ['shmem_SM_'+test_type]
        csv_writer.writerow(header)
    # Test different sizes
    for input_size in range(input_size_begin, input_size_end+1):
        run_test(input_size, device_id, test_type, occupancy, repeats, memory_loc)

def parse_commandline_argument():
    import argparse
    parser = argparse.ArgumentParser(description='Benchmark GUPS. Store results in results.csv file.')

    parser.add_argument("--device-id", help="GPU ID to run the test", default="0")
    parser.add_argument("--input-size-begin", help="exponent of the input data size begin range, base is 2 (input size = 2^n). "\
                        "[Default: 29 for global GUPS, max_shmem for shared GUPS. Global/shared is controlled by --memory-loc", default="0")
    parser.add_argument("--input-size-end", help="exponent of the input data size end range, base is 2 (input size = 2^n). "\
                        "[Default: 29 for global GUPS, max_shmem for shared GUPS. Global/shared is controlled by --memory-loc", default="0")
    parser.add_argument("--occupancy", help="100/occupancy is how much larger the working set is compared to the requested bytes", default="100")
    parser.add_argument("--repeats", help="number of kernel repetitions", default="1")
    parser.add_argument("--test", help="test to run",
                        choices=["reads", "writes", "reads_writes", "updates", "updates_no_loop", "all"], 
                        default="all")
    parser.add_argument("--memory-loc", help="memory buffer in global memory or shared memory",
                        choices=["global", "shared"], default="global")

    return parser.parse_args()

if __name__ == '__main__':
    main()
