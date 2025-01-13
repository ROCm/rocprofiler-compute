import os
import shutil
import subprocess
import pandas as pd
import pytest
import test_utils

config = {}
config["cleanup"] = True if "PYTEST_XDIST_WORKER_COUNT" in os.environ else False

indirs = [
    "tests/workloads/vcopy/MI100",
    "tests/workloads/vcopy/MI200",
    "tests/workloads/vcopy/MI300A_A1",
    "tests/workloads/vcopy/MI300X_A1",
]


@pytest.mark.misc
def test_valid_path():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        process = subprocess.run(
            [
                "build/rocprof-compute.bin",
                "analyze",
                "--path",
                workload_dir,
            ],
            capture_output=True,
            text=True,
        )
        assert process.returncode == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.misc
def test_list_kernels():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        process = subprocess.run(
            [
                "build/rocprof-compute.bin",
                "analyze",
                "--path",
                workload_dir,
                "--list-stats",
            ],
            capture_output=True,
            text=True,
        )
        assert process.returncode == 0
    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.list_metrics
def test_list_metrics_gfx90a():
    process = subprocess.run(
        ["build/rocprof-compute.bin", "analyze", "--list-metrics", "gfx90a"],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 1

    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        process = subprocess.run(
            [
                "build/rocprof-compute.bin",
                "analyze",
                "--path",
                workload_dir,
                "--list-metrics",
                "gfx90a",
            ],
            capture_output=True,
            text=True,
        )
        assert process.returncode == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.list_metrics
def test_list_metrics_gfx906():
    process = subprocess.run(
        ["build/rocprof-compute.bin", "analyze", "--list-metrics", "gfx906"],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 1

    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        process = subprocess.run(
            [
                "build/rocprof-compute.bin",
                "analyze",
                "--path",
                workload_dir,
                "--list-metrics",
                "gfx906",
            ],
            capture_output=True,
            text=True,
        )
        assert process.returncode == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.list_metrics
def test_list_metrics_gfx908():
    process = subprocess.run(
        ["build/rocprof-compute.bin", "analyze", "--list-metrics", "gfx908"],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 1

    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        process = subprocess.run(
            [
                "build/rocprof-compute.bin",
                "analyze",
                "--path",
                workload_dir,
                "--list-metrics",
                "gfx908",
            ],
            capture_output=True,
            text=True,
        )
        assert process.returncode == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.filter_block
def test_filter_block_1():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        process = subprocess.run(
            [
                "build/rocprof-compute.bin",
                "analyze",
                "--path",
                workload_dir,
                "--block",
                "1",
            ],
            capture_output=True,
            text=True,
        )
        assert process.returncode == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.filter_block
def test_filter_block_2():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        process = subprocess.run(
            [
                "build/rocprof-compute.bin",
                "analyze",
                "--path",
                workload_dir,
                "--block",
                "5",
            ],
            capture_output=True,
            text=True,
        )
        assert process.returncode == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.filter_block
def test_filter_block_3():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        process = subprocess.run(
            [
                "build/rocprof-compute.bin",
                "analyze",
                "--path",
                workload_dir,
                "--block",
                "5.2.2",
            ],
            capture_output=True,
            text=True,
        )
        assert process.returncode == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.filter_block
def test_filter_block_4():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        process = subprocess.run(
            [
                "build/rocprof-compute.bin",
                "analyze",
                "--path",
                workload_dir,
                "--block",
                "6.1",
            ],
            capture_output=True,
            text=True,
        )
        assert process.returncode == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.filter_block
def test_filter_block_5():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        process = subprocess.run(
            [
                "build/rocprof-compute.bin",
                "analyze",
                "--path",
                workload_dir,
                "--block",
                "10",
            ],
            capture_output=True,
            text=True,
        )
        assert process.returncode == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.filter_block
def test_filter_block_6():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        process = subprocess.run(
            [
                "build/rocprof-compute.bin",
                "analyze",
                "--path",
                workload_dir,
                "--block",
                "100",
            ],
            capture_output=True,
            text=True,
        )
        assert process.returncode == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.serial
def test_filter_kernel_1():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        process = subprocess.run(
            [
                "build/rocprof-compute.bin",
                "analyze",
                "--path",
                workload_dir,
                "--kernel",
                "0",
            ],
            capture_output=True,
            text=True,
        )
        assert process.returncode == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.serial
def test_filter_kernel_2():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        process = subprocess.run(
            [
                "build/rocprof-compute.bin",
                "analyze",
                "--path",
                workload_dir,
                "--kernel",
                "1",
            ],
            capture_output=True,
            text=True,
        )
        assert process.returncode == 1

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.serial
def test_filter_kernel_3():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        process = subprocess.run(
            [
                "build/rocprof-compute.bin",
                "analyze",
                "--path",
                workload_dir,
                "--kernel",
                "0",
                "1",
            ],
            capture_output=True,
            text=True,
        )
        assert process.returncode == 1

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.serial
def test_dispatch_1():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        process = subprocess.run(
            [
                "build/rocprof-compute.bin",
                "analyze",
                "--path",
                workload_dir,
                "--dispatch",
                "0",
            ],
            capture_output=True,
            text=True,
        )
        assert process.returncode == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.serial
def test_dispatch_2():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        process = subprocess.run(
            [
                "build/rocprof-compute.bin",
                "analyze",
                "--path",
                workload_dir,
                "--dispatch",
                "1",
            ],
            capture_output=True,
            text=True,
        )
        assert process.returncode == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.serial
def test_dispatch_3():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        process = subprocess.run(
            [
                "build/rocprof-compute.bin",
                "analyze",
                "--path",
                workload_dir,
                "--dispatch",
                "2",
            ],
            capture_output=True,
            text=True,
        )
        assert process.returncode == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.serial
def test_dispatch_4():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        process = subprocess.run(
            [
                "build/rocprof-compute.bin",
                "analyze",
                "--path",
                workload_dir,
                "--dispatch",
                "1",
                "4",
            ],
            capture_output=True,
            text=True,
        )
        assert process.returncode == 1

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.serial
def test_dispatch_5():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        process = subprocess.run(
            [
                "build/rocprof-compute.bin",
                "analyze",
                "--path",
                workload_dir,
                "--dispatch",
                "5",
                "6",
            ],
            capture_output=True,
            text=True,
        )
        assert process.returncode == 1

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.misc
def test_gpu_ids():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        process = subprocess.run(
            [
                "build/rocprof-compute.bin",
                "analyze",
                "--path",
                workload_dir,
                "--gpu-id",
                "2",
            ],
            capture_output=True,
            text=True,
        )
        assert process.returncode == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.normal_unit
def test_normal_unit_per_wave():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        process = subprocess.run(
            [
                "build/rocprof-compute.bin",
                "analyze",
                "--path",
                workload_dir,
                "--normal-unit",
                "per_wave",
            ],
            capture_output=True,
            text=True,
        )
        assert process.returncode == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.normal_unit
def test_normal_unit_per_cycle():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        process = subprocess.run(
            [
                "build/rocprof-compute.bin",
                "analyze",
                "--path",
                workload_dir,
                "--normal-unit",
                "per_cycle",
            ],
            capture_output=True,
            text=True,
        )
        assert process.returncode == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.normal_unit
def test_normal_unit_per_second():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        process = subprocess.run(
            [
                "build/rocprof-compute.bin",
                "analyze",
                "--path",
                workload_dir,
                "--normal-unit",
                "per_second",
            ],
            capture_output=True,
            text=True,
        )
        assert process.returncode == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.normal_unit
def test_normal_unit_per_kernel():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        process = subprocess.run(
            [
                "build/rocprof-compute.bin",
                "analyze",
                "--path",
                workload_dir,
                "--normal-unit",
                "per_kernel",
            ],
            capture_output=True,
            text=True,
        )
        assert process.returncode == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.max_stat
def test_max_stat_num_1():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        process = subprocess.run(
            [
                "build/rocprof-compute.bin",
                "analyze",
                "--path",
                workload_dir,
                "--max-stat-num",
                "0",
            ],
            capture_output=True,
            text=True,
        )
        assert process.returncode == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.max_stat
def test_max_stat_num_2():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        process = subprocess.run(
            [
                "build/rocprof-compute.bin",
                "analyze",
                "--path",
                workload_dir,
                "--max-stat-num",
                "5",
            ],
            capture_output=True,
            text=True,
        )
        assert process.returncode == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.max_stat
def test_max_stat_num_3():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        process = subprocess.run(
            [
                "build/rocprof-compute.bin",
                "analyze",
                "--path",
                workload_dir,
                "--max-stat-num",
                "10",
            ],
            capture_output=True,
            text=True,
        )
        assert process.returncode == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.max_stat
def test_max_stat_num_4():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        process = subprocess.run(
            [
                "build/rocprof-compute.bin",
                "analyze",
                "--path",
                workload_dir,
                "--max-stat-num",
                "15",
            ],
            capture_output=True,
            text=True,
        )
        assert process.returncode == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.time_unit
def test_time_unit_s():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        process = subprocess.run(
            [
                "build/rocprof-compute.bin",
                "analyze",
                "--path",
                workload_dir,
                "--time-unit",
                "s",
            ],
            capture_output=True,
            text=True,
        )
        assert process.returncode == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.time_unit
def test_time_unit_ms():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        process = subprocess.run(
            [
                "build/rocprof-compute.bin",
                "analyze",
                "--path",
                workload_dir,
                "--time-unit",
                "ms",
            ],
            capture_output=True,
            text=True,
        )
        assert process.returncode == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.time_unit
def test_time_unit_us():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        process = subprocess.run(
            [
                "build/rocprof-compute.bin",
                "analyze",
                "--path",
                workload_dir,
                "--time-unit",
                "us",
            ],
            capture_output=True,
            text=True,
        )
        assert process.returncode == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.time_unit
def test_time_unit_ns():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        process = subprocess.run(
            [
                "build/rocprof-compute.bin",
                "analyze",
                "--path",
                workload_dir,
                "--time-unit",
                "ns",
            ],
            capture_output=True,
            text=True,
        )
        assert process.returncode == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.decimal
def test_decimal_1():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        process = subprocess.run(
            [
                "build/rocprof-compute.bin",
                "analyze",
                "--path",
                workload_dir,
                "--decimal",
                "0",
            ],
            capture_output=True,
            text=True,
        )
        assert process.returncode == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.decimal
def test_decimal_2():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        process = subprocess.run(
            [
                "build/rocprof-compute.bin",
                "analyze",
                "--path",
                workload_dir,
                "--decimal",
                "1",
            ],
            capture_output=True,
            text=True,
        )
        assert process.returncode == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.decimal
def test_decimal_3():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        process = subprocess.run(
            [
                "build/rocprof-compute.bin",
                "analyze",
                "--path",
                workload_dir,
                "--decimal",
                "4",
            ],
            capture_output=True,
            text=True,
        )
        assert process.returncode == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.misc
def test_save_dfs():
    output_path = "tests/workloads/vcopy/saved_analysis"
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        process = subprocess.run(
            [
                "build/rocprof-compute.bin",
                "analyze",
                "--path",
                workload_dir,
                "--save-dfs",
                output_path,
            ],
            capture_output=True,
            text=True,
        )
        assert process.returncode == 0

        files_in_workload = os.listdir(output_path)
        single_row_tables = [
            "0.1_Top_Kernels.csv",
            "13.3_Instruction_Cache_-_L2_Interface.csv",
            "18.1_Aggregate_Stats_(All_channels).csv",
        ]
        for file_name in files_in_workload:
            df = pd.read_csv(output_path + "/" + file_name)
            if file_name in single_row_tables:
                assert len(df.index) == 1
            else:
                assert len(df.index) >= 3

        shutil.rmtree(output_path)
    test_utils.clean_output_dir(config["cleanup"], workload_dir)

    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        process = subprocess.run(
            [
                "build/rocprof-compute.bin",
                "analyze",
                "--path",
                workload_dir,
                "--save-dfs",
                output_path,
            ],
            capture_output=True,
            text=True,
        )
    assert process.returncode == 0

    files_in_workload = os.listdir(output_path)
    for file_name in files_in_workload:
        df = pd.read_csv(output_path + "/" + file_name)
        if file_name in single_row_tables:
            assert len(df.index) == 1
        else:
            assert len(df.index) >= 3
    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.col
def test_col_1():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        process = subprocess.run(
            [
                "build/rocprof-compute.bin",
                "analyze",
                "--path",
                workload_dir,
                "--cols",
                "0",
            ],
            capture_output=True,
            text=True,
        )
        assert process.returncode == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.col
def test_col_2():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        process = subprocess.run(
            [
                "build/rocprof-compute.bin",
                "analyze",
                "--path",
                workload_dir,
                "--cols",
                "2",
            ],
            capture_output=True,
            text=True,
        )
        assert process.returncode == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.col
def test_col_3():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        process = subprocess.run(
            [
                "build/rocprof-compute.bin",
                "analyze",
                "--path",
                workload_dir,
                "--cols",
                "0",
                "2",
            ],
            capture_output=True,
            text=True,
        )
        assert process.returncode == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.misc
def test_g():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        process = subprocess.run(
            [
                "build/rocprof-compute.bin",
                "analyze",
                "--path",
                workload_dir,
                "-g",
            ],
            capture_output=True,
            text=True,
        )
        assert process.returncode == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.kernel_verbose
def test_kernel_verbose_0():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        process = subprocess.run(
            [
                "build/rocprof-compute.bin",
                "analyze",
                "--path",
                workload_dir,
                "--kernel-verbose",
                "0",
            ],
            capture_output=True,
            text=True,
        )
        assert process.returncode == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.kernel_verbose
def test_kernel_verbose_1():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        process = subprocess.run(
            [
                "build/rocprof-compute.bin",
                "analyze",
                "--path",
                workload_dir,
                "--kernel-verbose",
                "1",
            ],
            capture_output=True,
            text=True,
        )
        assert process.returncode == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.kernel_verbose
def test_kernel_verbose_2():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        process = subprocess.run(
            [
                "build/rocprof-compute.bin",
                "analyze",
                "--path",
                workload_dir,
                "--kernel-verbose",
                "2",
            ],
            capture_output=True,
            text=True,
        )
        assert process.returncode == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.kernel_verbose
def test_kernel_verbose_3():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        process = subprocess.run(
            [
                "build/rocprof-compute.bin",
                "analyze",
                "--path",
                workload_dir,
                "--kernel-verbose",
                "3",
            ],
            capture_output=True,
            text=True,
        )
        assert process.returncode == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.kernel_verbose
def test_kernel_verbose_4():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        process = subprocess.run(
            [
                "build/rocprof-compute.bin",
                "analyze",
                "--path",
                workload_dir,
                "--kernel-verbose",
                "4",
            ],
            capture_output=True,
            text=True,
        )
        assert process.returncode == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.kernel_verbose
def test_kernel_verbose_5():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        process = subprocess.run(
            [
                "build/rocprof-compute.bin",
                "analyze",
                "--path",
                workload_dir,
                "--kernel-verbose",
                "5",
            ],
            capture_output=True,
            text=True,
        )
        assert process.returncode == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.kernel_verbose
def test_kernel_verbose_6():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        process = subprocess.run(
            [
                "build/rocprof-compute.bin",
                "analyze",
                "--path",
                workload_dir,
                "--kernel-verbose",
                "6",
            ],
            capture_output=True,
            text=True,
        )
        assert process.returncode == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.misc
def test_baseline():
    process = subprocess.run(
        [
            "build/rocprof-compute.bin",
            "analyze",
            "--path",
            "tests/workloads/vcopy/MI200",
            "--path",
            "tests/workloads/vcopy/MI100",
        ],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0

    process = subprocess.run(
        [
            "build/rocprof-compute.bin",
            "analyze",
            "--path",
            "tests/workloads/vcopy/MI200",
            "--path",
            "tests/workloads/vcopy/MI200",
        ],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 1

    process = subprocess.run(
        [
            "build/rocprof-compute.bin",
            "analyze",
            "--path",
            "tests/workloads/vcopy/MI100",
            "--path",
            "tests/workloads/vcopy/MI100",
        ],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 1


@pytest.mark.misc
def test_dependency_MI100():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        process = subprocess.run(
            [
                "build/rocprof-compute.bin",
                "analyze",
                "--path",
                workload_dir,
                "--dependency",
            ],
            capture_output=True,
            text=True,
        )
        assert process.returncode == 0
    test_utils.clean_output_dir(config["cleanup"], workload_dir)
