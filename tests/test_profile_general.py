import os.path
from pathlib import Path
from unittest.mock import patch
import pytest
from importlib.machinery import SourceFileLoader
import pandas as pd
import subprocess
import re
import shutil
import inspect
import sys

omniperf = SourceFileLoader("omniperf", "src/omniperf").load_module()
workload_1 = os.path.realpath("workload")
kernel_name_1 = "vecCopy(double*, double*, double*, int, int) [clone .kd]"

app_1 = ["./tests/vcopy", "-n", "1048576", "-b", "256", "-i", "3"]
num_kernels = 3

dispatch_id = 0

DEFAULT_ABS_DIFF = 2.2
DEFAULT_REL_DIFF = 8
MAX_METRIC_VIOLATIONS = 2

ALL_CSVS = [
    "SQ_IFETCH_LEVEL.csv",
    "SQ_INST_LEVEL_LDS.csv",
    "SQ_INST_LEVEL_SMEM.csv",
    "SQ_INST_LEVEL_VMEM.csv",
    "SQ_LEVEL_WAVES.csv",
    "pmc_perf.csv",
    "pmc_perf_0.csv",
    "pmc_perf_1.csv",
    "pmc_perf_10.csv",
    "pmc_perf_11.csv",
    "pmc_perf_12.csv",
    "pmc_perf_13.csv",
    "pmc_perf_14.csv",
    "pmc_perf_15.csv",
    "pmc_perf_16.csv",
    "pmc_perf_2.csv",
    "pmc_perf_3.csv",
    "pmc_perf_4.csv",
    "pmc_perf_5.csv",
    "pmc_perf_6.csv",
    "pmc_perf_7.csv",
    "pmc_perf_8.csv",
    "pmc_perf_9.csv",
    "sysinfo.csv",
    "timestamps.csv",
]
ALL_CSVS_MI200 = [
    "SQ_IFETCH_LEVEL.csv",
    "SQ_INST_LEVEL_LDS.csv",
    "SQ_INST_LEVEL_SMEM.csv",
    "SQ_INST_LEVEL_VMEM.csv",
    "SQ_LEVEL_WAVES.csv",
    "pmc_perf.csv",
    "pmc_perf_0.csv",
    "pmc_perf_1.csv",
    "pmc_perf_10.csv",
    "pmc_perf_11.csv",
    "pmc_perf_12.csv",
    "pmc_perf_13.csv",
    "pmc_perf_14.csv",
    "pmc_perf_15.csv",
    "pmc_perf_16.csv",
    "pmc_perf_2.csv",
    "pmc_perf_3.csv",
    "pmc_perf_4.csv",
    "pmc_perf_5.csv",
    "pmc_perf_6.csv",
    "pmc_perf_7.csv",
    "pmc_perf_8.csv",
    "pmc_perf_9.csv",
    "roofline.csv",
    "sysinfo.csv",
    "timestamps.csv",
]
ROOF_ONLY_FILES = [
    "empirRoof_gpu-ALL_fp32.pdf",
    "empirRoof_gpu-ALL_int8_fp16.pdf",
    "pmc_perf.csv",
    "pmc_perf_0.csv",
    "pmc_perf_1.csv",
    "pmc_perf_2.csv",
    "roofline.csv",
    "sysinfo.csv",
    "timestamps.csv",
]

# logging function for threshold outliers set to false
COUNTER_LOGGING = False
METRIC_LOGGING = True

# Absolute Difference < 2
CONSISTENT_ABS_METRIC_INDICES = [
    "2.1.8",
    # "2.1.28",
    "2.1.9",
    "5.1.3",
    "13.1.0",
    "14.1.0",
    "5.2.4",
    "11.2.0",
    "11.2.2",
    "11.2.3",
    "13.1.0",
    "13.2.3",
    "16.5.3",
    "17.2.3",
    "17.2.8",
    "17.2.9",
    "17.3.8",
    "17.3.9",
    "17.3.11",
    "17.3.14",
    "17.3.15",
    "17.3.16",
    "18.1.13",
    "18.1.16",
]
# Percent Difference < 8
CONSISTENT_REL_METRIC_INDICES = [
    "2.1.26",
    # "5.1.0",
    # "5.1.1",
    # "5.2.0",
    # "5.2.1",
    "5.2.3",
    # "5.2.4",
    "5.2.6",
    "5.2.8",
    # "6.1.0",
    # "6.1.1",
    "6.1.3" "6.1.6",
    # "6.1.7",
    # "6.2.0",
    # "7.2.1",
    # "7.2.3",
    "7.2.4",
    "7.2.7",
    # "14.1.0",
    "16.2.0",
    # "16.3.14",
    # "16.3.15",
    "17.1.0",
]


def metric_compare(test_name, errors_pd, baseline_df, run_df, threshold=5):
    # iterate data one row at a time
    for idx_1 in run_df.index:
        run_row = run_df.iloc[idx_1]
        baseline_row = baseline_df.iloc[idx_1]
        if not run_row["KernelName"] == baseline_row["KernelName"]:
            print("Kernel/dispatch mismatch")
            assert 0
        kernel_name = run_row["KernelName"]
        gpu_id = run_row["gpu-id"]
        differences = {}

        for pmc_counter in run_row.index:
            if "Ns" in pmc_counter or "id" in pmc_counter or "[" in pmc_counter:
                # print("skipping "+pmc_counter)
                continue
                # assert 0

            if not pmc_counter in list(baseline_df.columns):
                print("error: pmc mismatch! " + pmc_counter + " is not in baseline_df")
                continue

            run_data = run_row[pmc_counter]
            baseline_data = baseline_row[pmc_counter]
            if isinstance(run_data, str) and isinstance(baseline_data, str):
                if run_data not in baseline_data:
                    print(baseline_data)
            else:
                # relative difference
                if not run_data == 0:
                    diff = round(100 * abs(baseline_data - run_data) / run_data, 2)
                    if diff > threshold:
                        print("[" + pmc_counter + "] diff is :" + str(diff) + "%")
                        if pmc_counter not in differences.keys():
                            print(
                                "[" + pmc_counter + "] not found in ",
                                list(differences.keys()),
                            )
                            differences[pmc_counter] = [diff]
                        else:
                            # Why are we here?
                            print(
                                "Why did we get here?!?!? errors_pd[idx_1]:",
                                list(differences.keys()),
                            )
                            differences[pmc_counter].append(diff)
                else:
                    # if 0 show absolute difference
                    diff = round(baseline_data - run_data, 2)
                    if diff > threshold:
                        print(str(idx_1) + "[" + pmc_counter + "] diff is :" + str(diff))
        differences["kernel_name"] = [kernel_name]
        differences["test_name"] = [test_name]
        differences["gpu-id"] = [gpu_id]
        errors_pd = pd.concat([errors_pd, pd.DataFrame.from_dict(differences)])
    return errors_pd


def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if cmd[0] == "rocm-smi" and p.returncode == 8:
        print("ERROR: No GPU detected. Unable to load rocm-smi")
        assert 0
    return p.stdout.decode("ascii")


def gpu_soc():
    rocminfo = str(
        subprocess.run(
            ["rocminfo"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ).stdout.decode("ascii")
    )
    rocminfo = rocminfo.split("\n")
    soc_regex = re.compile(r"^\s*Name\s*:\s+ ([a-zA-Z0-9]+)\s*$", re.MULTILINE)
    gpu_id = list(filter(soc_regex.match, rocminfo))[0].split()[1]

    if gpu_id == "gfx906":
        return "mi50"
    elif gpu_id == "gfx908":
        return "mi100"
    elif gpu_id == "gfx90a":
        return "mi200"
    elif gpu_id == "gfx900":
        return "vega10"
    else:
        print("Invalid SoC")
        assert 0


soc = gpu_soc()

# change to directory where baseline is at
Baseline_dir = os.path.realpath("Baseline_vcopy_" + soc)
if os.path.exists(Baseline_dir):
    shutil.rmtree(Baseline_dir)
with pytest.raises(SystemExit) as e:
    with patch(
        "sys.argv",
        [
            "omniperf",
            "profile",
            "-n",
            "app_1",
            "-VVV",
            "--path",
            Baseline_dir,
            "--",
        ]
        + app_1,
    ):
        omniperf.main()


def log_counter(file_dict, test_name):
    for file in file_dict.keys():
        if file == "pmc_perf.csv" or "SQ" in file:
            # read file in Baseline
            df_1 = pd.read_csv(Baseline_dir + "/" + file, index_col=0)
            # get corresponding file from current test run
            df_2 = file_dict[file]

            errors = metric_compare(test_name, pd.DataFrame(), df_1, df_2, 5)
            if not errors.empty:
                if os.path.exists(
                    Baseline_dir + "/" + file.split(".")[0] + "_error_log.csv"
                ):
                    error_log = pd.read_csv(
                        Baseline_dir + "/" + file.split(".")[0] + "_error_log.csv",
                        index_col=0,
                    )
                    new_error_log = pd.concat([error_log, errors])
                    new_error_log = new_error_log.reindex(
                        sorted(new_error_log.columns), axis=1
                    )
                    new_error_log = new_error_log.sort_values(
                        by=["test_name", "kernel_name", "gpu-id"]
                    )
                    new_error_log.to_csv(
                        Baseline_dir + "/" + file.split(".")[0] + "_error_log.csv"
                    )
                else:
                    errors.to_csv(
                        Baseline_dir + "/" + file.split(".")[0] + "_error_log.csv"
                    )


def log_metric(test_name, thresholds, args=[]):
    t = subprocess.Popen(
        [
            sys.executable,
            "src/omniperf",
            "analyze",
            "--path",
            Baseline_dir,
        ]
        + args
        + ["--path", workload_1, "--report-diff", str(DEFAULT_REL_DIFF)],
        stdout=subprocess.PIPE,
    )
    captured_output = t.communicate(timeout=1300)[0].decode("utf-8")
    assert t.returncode == 0

    if "DEBUG ERROR" in captured_output:
        error_df = pd.DataFrame()
        if os.path.exists(Baseline_dir + "/metric_error_log.csv"):
            error_df = pd.read_csv(
                Baseline_dir + "/metric_error_log.csv",
                index_col=0,
            )
        output_metric_errors = re.findall(r"(\')([0-9.]*)(\')", captured_output)
        high_diff_metrics = [x[1] for x in output_metric_errors]
        for metric in high_diff_metrics:
            metric_info = re.findall(
                r"(^"
                + metric
                + r")(?: *)([()0-9A-Za-z- ]+ )(?: *)([0-9.-]*)(?: *)([0-9.-]*)(?: *)\(([-0-9.]*)%\)(?: *)([-0-9.e]*)",
                captured_output,
                flags=re.MULTILINE,
            )
            if len(metric_info):
                metric_info = metric_info[0]
                metric_idx = metric_info[0]
                table_idx = metric_info[0].split(".")[0]
                relative_diff = float(metric_info[-2])
                absolute_diff = float(metric_info[-1])
                if relative_diff > -99 or relative_diff < -101:
                    relative_threshold = thresholds["default"]["relative"]
                    absolute_threshold = thresholds["default"]["absolute"]

                    if table_idx in thresholds:
                        relative_threshold = thresholds[table_idx]["relative"]
                        absolute_threshold = thresholds[table_idx]["absolute"]
                    if (
                        abs(relative_diff) > relative_threshold
                        and (metric_idx in CONSISTENT_REL_METRIC_INDICES)
                    ) or (
                        abs(absolute_diff) > absolute_threshold
                        and (metric_idx in CONSISTENT_ABS_METRIC_INDICES)
                    ):
                        new_error = pd.DataFrame.from_dict(
                            {
                                "Index": [metric_info[0]],
                                "Metric": [metric_info[1].strip()],
                                "Percent Difference": [relative_diff],
                                "Absolute Difference": [absolute_diff],
                                "Baseline": [metric_info[-3]],
                                "Current": [metric_info[-4]],
                                "Test Name": [test_name],
                            }
                        )
                        error_df = pd.concat([error_df, new_error])
                        counts = error_df.groupby("Index").cumcount()
                        failed_metrics = error_df.loc[counts > MAX_METRIC_VIOLATIONS]
                        if failed_metrics.any(axis=None):
                            print(
                                "Warning, these metrics are varying too much",
                                failed_metrics,
                            )

        if not error_df.empty:
            error_df.to_csv(Baseline_dir + "/metric_error_log.csv")


def test_path():
    if os.path.exists(workload_1):
        shutil.rmtree(workload_1)
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--",
            ]
            + app_1,
        ):
            omniperf.main()
    # assert successful run
    assert e.value.code == 0

    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1 + "/" + file, index_col=0)
            # TODO: verify contents: we know function evaluated
            if not "sysinfo" in file:
                assert len(file_dict[file].index) >= num_kernels
    
    if soc == "mi200":
        
        assert sorted(list(file_dict.keys())) == ALL_CSVS_MI200
    else:
        assert sorted(list(file_dict.keys())) == ALL_CSVS

    if COUNTER_LOGGING:
        log_counter(file_dict, inspect.stack()[0][3])

    if METRIC_LOGGING:
        log_metric(
            inspect.stack()[0][3],
            {"default": {"absolute": DEFAULT_ABS_DIFF, "relative": DEFAULT_REL_DIFF}},
        )


def test_kernel():
    if os.path.exists(workload_1):
        shutil.rmtree(workload_1)
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--kernel",
                kernel_name_1,
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0

    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1 + "/" + file)
            if not "sysinfo" in file:
                assert len(file_dict[file].index) >= num_kernels
    if soc == "mi200":
        
        assert sorted(list(file_dict.keys())) == ALL_CSVS_MI200
    else:
        assert sorted(list(file_dict.keys())) == ALL_CSVS

    if COUNTER_LOGGING:
        log_counter(file_dict, inspect.stack()[0][3])

    if METRIC_LOGGING:
        log_metric(
            inspect.stack()[0][3],
            {"default": {"absolute": DEFAULT_ABS_DIFF, "relative": DEFAULT_REL_DIFF}},
        )


def test_kernel_summaries():
    if os.path.exists(workload_1):
        shutil.rmtree(workload_1)
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--kernel-summaries",
                "vcopy",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0

    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1 + "/" + file)
            if not "sysinfo" in file:
                assert len(file_dict[file].index) >= num_kernels
    if soc == "mi200":
        
        assert sorted(list(file_dict.keys())) == ALL_CSVS_MI200
    else:
        assert sorted(list(file_dict.keys())) == ALL_CSVS

    if COUNTER_LOGGING:
        log_counter(file_dict, inspect.stack()[0][3])

    if METRIC_LOGGING:
        log_metric(
            inspect.stack()[0][3],
            {"default": {"absolute": DEFAULT_ABS_DIFF, "relative": DEFAULT_REL_DIFF}},
        )


def test_ipblocks_SQ():
    if os.path.exists(workload_1):
        shutil.rmtree(workload_1)
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--ipblocks",
                "SQ",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0

    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1 + "/" + file)
            if not "sysinfo" in file:
                assert len(file_dict[file].index) >= num_kernels
    
    expected_csvs = [
        "SQ_IFETCH_LEVEL.csv",
        "SQ_INST_LEVEL_LDS.csv",
        "SQ_INST_LEVEL_SMEM.csv",
        "SQ_INST_LEVEL_VMEM.csv",
        "SQ_LEVEL_WAVES.csv",
        "pmc_perf.csv",
        "pmc_perf_0.csv",
        "pmc_perf_1.csv",
        "pmc_perf_2.csv",
        "pmc_perf_3.csv",
        "pmc_perf_4.csv",
        "pmc_perf_5.csv",
        "pmc_perf_6.csv",
        "pmc_perf_7.csv",
        "pmc_perf_8.csv",
        "sysinfo.csv",
        "timestamps.csv",
    ]
    if soc == "mi200":
        expected_csvs = [
            "SQ_IFETCH_LEVEL.csv",
            "SQ_INST_LEVEL_LDS.csv",
            "SQ_INST_LEVEL_SMEM.csv",
            "SQ_INST_LEVEL_VMEM.csv",
            "SQ_LEVEL_WAVES.csv",
            "pmc_perf.csv",
            "pmc_perf_0.csv",
            "pmc_perf_1.csv",
            "pmc_perf_10.csv",
            "pmc_perf_11.csv",
            "pmc_perf_12.csv",
            "pmc_perf_2.csv",
            "pmc_perf_3.csv",
            "pmc_perf_4.csv",
            "pmc_perf_5.csv",
            "pmc_perf_6.csv",
            "pmc_perf_7.csv",
            "pmc_perf_8.csv",
            "pmc_perf_9.csv",
            "roofline.csv",
            "sysinfo.csv",
            "timestamps.csv",
        ]

    assert sorted(list(file_dict.keys())) == expected_csvs

    if COUNTER_LOGGING:
        log_counter(file_dict, inspect.stack()[0][3])

    if METRIC_LOGGING:
        log_metric(
            inspect.stack()[0][3],
            {"default": {"absolute": DEFAULT_ABS_DIFF, "relative": DEFAULT_REL_DIFF}},
        )


def test_ipblocks_SQC():
    if os.path.exists(workload_1):
        shutil.rmtree(workload_1)
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--ipblocks",
                "SQC",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0

    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1 + "/" + file)
            if not "sysinfo" in file:
                assert len(file_dict[file].index) >= num_kernels
    
    expected_csvs = [
        "pmc_perf.csv",
        "pmc_perf_0.csv",
        "pmc_perf_1.csv",
        "pmc_perf_2.csv",
        "pmc_perf_3.csv",
        "sysinfo.csv",
        "timestamps.csv",
    ]
    if soc == "mi200":
        expected_csvs.insert(5, "roofline.csv")

    assert sorted(list(file_dict.keys())) == expected_csvs

    if COUNTER_LOGGING:
        log_counter(file_dict, inspect.stack()[0][3])

    if METRIC_LOGGING:
        log_metric(
            inspect.stack()[0][3],
            {"default": {"absolute": DEFAULT_ABS_DIFF, "relative": DEFAULT_REL_DIFF}},
        )


def test_ipblocks_TA():
    if os.path.exists(workload_1):
        shutil.rmtree(workload_1)
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--ipblocks",
                "TA",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0

    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1 + "/" + file)
            if not "sysinfo" in file:
                assert len(file_dict[file].index) >= num_kernels

    

    expected_csvs = [
        "pmc_perf.csv",
        "pmc_perf_0.csv",
        "pmc_perf_1.csv",
        "pmc_perf_2.csv",
        "pmc_perf_3.csv",
        "pmc_perf_4.csv",
        "pmc_perf_5.csv",
        "pmc_perf_6.csv",
        "pmc_perf_7.csv",
        "sysinfo.csv",
        "timestamps.csv",
    ]
    if soc == "mi200":
        expected_csvs.insert(9, "roofline.csv")

    assert sorted(list(file_dict.keys())) == expected_csvs

    if COUNTER_LOGGING:
        log_counter(file_dict, inspect.stack()[0][3])

    if METRIC_LOGGING:
        log_metric(
            inspect.stack()[0][3],
            {"default": {"absolute": DEFAULT_ABS_DIFF, "relative": DEFAULT_REL_DIFF}},
        )


def test_ipblocks_TD():
    if os.path.exists(workload_1):
        shutil.rmtree(workload_1)
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--ipblocks",
                "TD",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0

    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1 + "/" + file)
            if not "sysinfo" in file:
                assert len(file_dict[file].index) >= num_kernels

    

    expected_csvs = [
        "pmc_perf.csv",
        "pmc_perf_0.csv",
        "pmc_perf_1.csv",
        "pmc_perf_2.csv",
        "sysinfo.csv",
        "timestamps.csv",
    ]
    if soc == "mi200":
        expected_csvs = [
            "pmc_perf.csv",
            "pmc_perf_0.csv",
            "pmc_perf_1.csv",
            "pmc_perf_2.csv",
            "pmc_perf_3.csv",
            "roofline.csv",
            "sysinfo.csv",
            "timestamps.csv",
        ]

    assert sorted(list(file_dict.keys())) == expected_csvs

    if COUNTER_LOGGING:
        log_counter(file_dict, inspect.stack()[0][3])

    if METRIC_LOGGING:
        log_metric(
            inspect.stack()[0][3],
            {"default": {"absolute": DEFAULT_ABS_DIFF, "relative": DEFAULT_REL_DIFF}},
        )


def test_ipblocks_TCP():
    if os.path.exists(workload_1):
        shutil.rmtree(workload_1)
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--ipblocks",
                "TCP",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0

    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1 + "/" + file)
            if not "sysinfo" in file:
                assert len(file_dict[file].index) >= num_kernels
    

    expected_csvs = [
        "pmc_perf.csv",
        "pmc_perf_0.csv",
        "pmc_perf_1.csv",
        "pmc_perf_2.csv",
        "pmc_perf_3.csv",
        "pmc_perf_4.csv",
        "pmc_perf_5.csv",
        "pmc_perf_6.csv",
        "pmc_perf_7.csv",
        "pmc_perf_8.csv",
        "pmc_perf_9.csv",
        "sysinfo.csv",
        "timestamps.csv",
    ]
    if soc == "mi200":
        expected_csvs.insert(11, "roofline.csv")

    assert sorted(list(file_dict.keys())) == expected_csvs

    if COUNTER_LOGGING:
        log_counter(file_dict, inspect.stack()[0][3])

    if METRIC_LOGGING:
        log_metric(
            inspect.stack()[0][3],
            {"default": {"absolute": DEFAULT_ABS_DIFF, "relative": DEFAULT_REL_DIFF}},
        )


def test_ipblocks_TCC():
    if os.path.exists(workload_1):
        shutil.rmtree(workload_1)
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--ipblocks",
                "TCC",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0

    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1 + "/" + file)
            if not "sysinfo" in file:
                assert len(file_dict[file].index) >= num_kernels

    
    expected_csvs = [
        "pmc_perf.csv",
        "pmc_perf_0.csv",
        "pmc_perf_1.csv",
        "pmc_perf_10.csv",
        "pmc_perf_2.csv",
        "pmc_perf_3.csv",
        "pmc_perf_4.csv",
        "pmc_perf_5.csv",
        "pmc_perf_6.csv",
        "pmc_perf_7.csv",
        "pmc_perf_8.csv",
        "pmc_perf_9.csv",
        "sysinfo.csv",
        "timestamps.csv",
    ]
    if soc == "mi200":
        expected_csvs.insert(12, "roofline.csv")

    assert sorted(list(file_dict.keys())) == expected_csvs

    if COUNTER_LOGGING:
        log_counter(file_dict, inspect.stack()[0][3])

    if METRIC_LOGGING:
        log_metric(
            inspect.stack()[0][3],
            {"default": {"absolute": DEFAULT_ABS_DIFF, "relative": DEFAULT_REL_DIFF}},
        )


def test_ipblocks_SPI():
    if os.path.exists(workload_1):
        shutil.rmtree(workload_1)
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--ipblocks",
                "SPI",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0

    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1 + "/" + file)
            if not "sysinfo" in file:
                assert len(file_dict[file].index) >= num_kernels
    

    expected_csvs = [
        "pmc_perf.csv",
        "pmc_perf_0.csv",
        "pmc_perf_1.csv",
        "pmc_perf_2.csv",
        "pmc_perf_3.csv",
        "pmc_perf_4.csv",
        "pmc_perf_5.csv",
        "pmc_perf_6.csv",
        "pmc_perf_7.csv",
        "pmc_perf_8.csv",
        "sysinfo.csv",
        "timestamps.csv",
    ]
    if soc == "mi200":
        expected_csvs.insert(10, "roofline.csv")

    assert sorted(list(file_dict.keys())) == expected_csvs

    if COUNTER_LOGGING:
        log_counter(file_dict, inspect.stack()[0][3])

    if METRIC_LOGGING:
        log_metric(
            inspect.stack()[0][3],
            {"default": {"absolute": DEFAULT_ABS_DIFF, "relative": DEFAULT_REL_DIFF}},
        )


def test_ipblocks_CPC():
    if os.path.exists(workload_1):
        shutil.rmtree(workload_1)
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--ipblocks",
                "CPC",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0

    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1 + "/" + file)
            if not "sysinfo" in file:
                assert len(file_dict[file].index) >= num_kernels
    
    expected_csvs = [
        "pmc_perf.csv",
        "pmc_perf_0.csv",
        "pmc_perf_1.csv",
        "pmc_perf_2.csv",
        "pmc_perf_3.csv",
        "pmc_perf_4.csv",
        "pmc_perf_5.csv",
        "sysinfo.csv",
        "timestamps.csv",
    ]
    if soc == "mi200":
        expected_csvs.insert(7, "roofline.csv")
    assert sorted(list(file_dict.keys())) == expected_csvs

    if COUNTER_LOGGING:
        log_counter(file_dict, inspect.stack()[0][3])

    if METRIC_LOGGING:
        log_metric(
            inspect.stack()[0][3],
            {"default": {"absolute": DEFAULT_ABS_DIFF, "relative": DEFAULT_REL_DIFF}},
        )


def test_ipblocks_CPF():
    if os.path.exists(workload_1):
        shutil.rmtree(workload_1)
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--ipblocks",
                "CPF",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0

    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1 + "/" + file)
            if not "sysinfo" in file:
                assert len(file_dict[file].index) >= num_kernels
    
    expected_csvs = [
        "pmc_perf.csv",
        "pmc_perf_0.csv",
        "pmc_perf_1.csv",
        "pmc_perf_2.csv",
        "pmc_perf_3.csv",
        "sysinfo.csv",
        "timestamps.csv",
    ]
    if soc == "mi200":
        expected_csvs.insert(5, "roofline.csv")
    assert sorted(list(file_dict.keys())) == expected_csvs

    if COUNTER_LOGGING:
        log_counter(file_dict, inspect.stack()[0][3])

    if METRIC_LOGGING:
        log_metric(
            inspect.stack()[0][3],
            {"default": {"absolute": DEFAULT_ABS_DIFF, "relative": DEFAULT_REL_DIFF}},
        )


def test_ipblocks_SQ_CPC():
    if os.path.exists(workload_1):
        shutil.rmtree(workload_1)
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--ipblocks",
                "SQ",
                "CPC",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0

    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1 + "/" + file)
            if not "sysinfo" in file:
                assert len(file_dict[file].index) >= num_kernels

    
    expected_csvs = [
        "SQ_IFETCH_LEVEL.csv",
        "SQ_INST_LEVEL_LDS.csv",
        "SQ_INST_LEVEL_SMEM.csv",
        "SQ_INST_LEVEL_VMEM.csv",
        "SQ_LEVEL_WAVES.csv",
        "pmc_perf.csv",
        "pmc_perf_0.csv",
        "pmc_perf_1.csv",
        "pmc_perf_2.csv",
        "pmc_perf_3.csv",
        "pmc_perf_4.csv",
        "pmc_perf_5.csv",
        "pmc_perf_6.csv",
        "pmc_perf_7.csv",
        "pmc_perf_8.csv",
        "sysinfo.csv",
        "timestamps.csv",
    ]
    if soc == "mi200":
        expected_csvs = [
            "SQ_IFETCH_LEVEL.csv",
            "SQ_INST_LEVEL_LDS.csv",
            "SQ_INST_LEVEL_SMEM.csv",
            "SQ_INST_LEVEL_VMEM.csv",
            "SQ_LEVEL_WAVES.csv",
            "pmc_perf.csv",
            "pmc_perf_0.csv",
            "pmc_perf_1.csv",
            "pmc_perf_10.csv",
            "pmc_perf_11.csv",
            "pmc_perf_12.csv",
            "pmc_perf_2.csv",
            "pmc_perf_3.csv",
            "pmc_perf_4.csv",
            "pmc_perf_5.csv",
            "pmc_perf_6.csv",
            "pmc_perf_7.csv",
            "pmc_perf_8.csv",
            "pmc_perf_9.csv",
            "roofline.csv",
            "sysinfo.csv",
            "timestamps.csv",
        ]

    assert sorted(list(file_dict.keys())) == expected_csvs

    if COUNTER_LOGGING:
        log_counter(file_dict, inspect.stack()[0][3])

    if METRIC_LOGGING:
        log_metric(
            inspect.stack()[0][3],
            {"default": {"absolute": DEFAULT_ABS_DIFF, "relative": DEFAULT_REL_DIFF}},
        )


def test_ipblocks_SQ_TA():
    if os.path.exists(workload_1):
        shutil.rmtree(workload_1)
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--ipblocks",
                "SQ",
                "TA",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0

    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1 + "/" + file)
            if not "sysinfo" in file:
                assert len(file_dict[file].index) >= num_kernels

    
    expected_csvs = [
        "SQ_IFETCH_LEVEL.csv",
        "SQ_INST_LEVEL_LDS.csv",
        "SQ_INST_LEVEL_SMEM.csv",
        "SQ_INST_LEVEL_VMEM.csv",
        "SQ_LEVEL_WAVES.csv",
        "pmc_perf.csv",
        "pmc_perf_0.csv",
        "pmc_perf_1.csv",
        "pmc_perf_2.csv",
        "pmc_perf_3.csv",
        "pmc_perf_4.csv",
        "pmc_perf_5.csv",
        "pmc_perf_6.csv",
        "pmc_perf_7.csv",
        "pmc_perf_8.csv",
        "sysinfo.csv",
        "timestamps.csv",
    ]
    if soc == "mi200":
        expected_csvs = [
            "SQ_IFETCH_LEVEL.csv",
            "SQ_INST_LEVEL_LDS.csv",
            "SQ_INST_LEVEL_SMEM.csv",
            "SQ_INST_LEVEL_VMEM.csv",
            "SQ_LEVEL_WAVES.csv",
            "pmc_perf.csv",
            "pmc_perf_0.csv",
            "pmc_perf_1.csv",
            "pmc_perf_10.csv",
            "pmc_perf_11.csv",
            "pmc_perf_12.csv",
            "pmc_perf_2.csv",
            "pmc_perf_3.csv",
            "pmc_perf_4.csv",
            "pmc_perf_5.csv",
            "pmc_perf_6.csv",
            "pmc_perf_7.csv",
            "pmc_perf_8.csv",
            "pmc_perf_9.csv",
            "roofline.csv",
            "sysinfo.csv",
            "timestamps.csv",
        ]
    assert sorted(list(file_dict.keys())) == expected_csvs

    if COUNTER_LOGGING:
        log_counter(file_dict, inspect.stack()[0][3])

    if METRIC_LOGGING:
        log_metric(
            inspect.stack()[0][3],
            {"default": {"absolute": DEFAULT_ABS_DIFF, "relative": DEFAULT_REL_DIFF}},
        )


def test_ipblocks_SQ_SPI():
    if os.path.exists(workload_1):
        shutil.rmtree(workload_1)
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--ipblocks",
                "SQ",
                "SPI",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0

    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1 + "/" + file)
            if not "sysinfo" in file:
                assert len(file_dict[file].index) >= num_kernels

    
    expected_csvs = [
        "SQ_IFETCH_LEVEL.csv",
        "SQ_INST_LEVEL_LDS.csv",
        "SQ_INST_LEVEL_SMEM.csv",
        "SQ_INST_LEVEL_VMEM.csv",
        "SQ_LEVEL_WAVES.csv",
        "pmc_perf.csv",
        "pmc_perf_0.csv",
        "pmc_perf_1.csv",
        "pmc_perf_2.csv",
        "pmc_perf_3.csv",
        "pmc_perf_4.csv",
        "pmc_perf_5.csv",
        "pmc_perf_6.csv",
        "pmc_perf_7.csv",
        "pmc_perf_8.csv",
        "sysinfo.csv",
        "timestamps.csv",
    ]
    if soc == "mi200":
        expected_csvs = [
            "SQ_IFETCH_LEVEL.csv",
            "SQ_INST_LEVEL_LDS.csv",
            "SQ_INST_LEVEL_SMEM.csv",
            "SQ_INST_LEVEL_VMEM.csv",
            "SQ_LEVEL_WAVES.csv",
            "pmc_perf.csv",
            "pmc_perf_0.csv",
            "pmc_perf_1.csv",
            "pmc_perf_10.csv",
            "pmc_perf_11.csv",
            "pmc_perf_12.csv",
            "pmc_perf_2.csv",
            "pmc_perf_3.csv",
            "pmc_perf_4.csv",
            "pmc_perf_5.csv",
            "pmc_perf_6.csv",
            "pmc_perf_7.csv",
            "pmc_perf_8.csv",
            "pmc_perf_9.csv",
            "roofline.csv",
            "sysinfo.csv",
            "timestamps.csv",
        ]
    assert sorted(list(file_dict.keys())) == expected_csvs

    if COUNTER_LOGGING:
        log_counter(file_dict, inspect.stack()[0][3])

    if METRIC_LOGGING:
        log_metric(
            inspect.stack()[0][3],
            {"default": {"absolute": DEFAULT_ABS_DIFF, "relative": DEFAULT_REL_DIFF}},
        )


def test_ipblocks_SQ_SQC_TCP_CPC():
    if os.path.exists(workload_1):
        shutil.rmtree(workload_1)
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--ipblocks",
                "SQ",
                "SQC",
                "TCP",
                "CPC",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0

    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1 + "/" + file)
            if not "sysinfo" in file:
                assert len(file_dict[file].index) >= num_kernels

    
    expected_csvs = [
        "SQ_IFETCH_LEVEL.csv",
        "SQ_INST_LEVEL_LDS.csv",
        "SQ_INST_LEVEL_SMEM.csv",
        "SQ_INST_LEVEL_VMEM.csv",
        "SQ_LEVEL_WAVES.csv",
        "pmc_perf.csv",
        "pmc_perf_0.csv",
        "pmc_perf_1.csv",
        "pmc_perf_2.csv",
        "pmc_perf_3.csv",
        "pmc_perf_4.csv",
        "pmc_perf_5.csv",
        "pmc_perf_6.csv",
        "pmc_perf_7.csv",
        "pmc_perf_8.csv",
        "pmc_perf_9.csv",
        "sysinfo.csv",
        "timestamps.csv",
    ]
    if soc == "mi200":
        expected_csvs = [
            "SQ_IFETCH_LEVEL.csv",
            "SQ_INST_LEVEL_LDS.csv",
            "SQ_INST_LEVEL_SMEM.csv",
            "SQ_INST_LEVEL_VMEM.csv",
            "SQ_LEVEL_WAVES.csv",
            "pmc_perf.csv",
            "pmc_perf_0.csv",
            "pmc_perf_1.csv",
            "pmc_perf_10.csv",
            "pmc_perf_11.csv",
            "pmc_perf_12.csv",
            "pmc_perf_2.csv",
            "pmc_perf_3.csv",
            "pmc_perf_4.csv",
            "pmc_perf_5.csv",
            "pmc_perf_6.csv",
            "pmc_perf_7.csv",
            "pmc_perf_8.csv",
            "pmc_perf_9.csv",
            "roofline.csv",
            "sysinfo.csv",
            "timestamps.csv",
        ]
    assert sorted(list(file_dict.keys())) == expected_csvs

    if COUNTER_LOGGING:
        log_counter(file_dict, inspect.stack()[0][3])

    if METRIC_LOGGING:
        log_metric(
            inspect.stack()[0][3],
            {"default": {"absolute": DEFAULT_ABS_DIFF, "relative": DEFAULT_REL_DIFF}},
        )


def test_ipblocks_SQ_SPI_TA_TCC_CPF():
    if os.path.exists(workload_1):
        shutil.rmtree(workload_1)
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--ipblocks",
                "SQ",
                "SPI",
                "TA",
                "TCC",
                "CPF",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0

    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1 + "/" + file)
            if not "sysinfo" in file:
                assert len(file_dict[file].index) >= num_kernels

    
    expected_csvs = [
        "SQ_IFETCH_LEVEL.csv",
        "SQ_INST_LEVEL_LDS.csv",
        "SQ_INST_LEVEL_SMEM.csv",
        "SQ_INST_LEVEL_VMEM.csv",
        "SQ_LEVEL_WAVES.csv",
        "pmc_perf.csv",
        "pmc_perf_0.csv",
        "pmc_perf_1.csv",
        "pmc_perf_10.csv",
        "pmc_perf_2.csv",
        "pmc_perf_3.csv",
        "pmc_perf_4.csv",
        "pmc_perf_5.csv",
        "pmc_perf_6.csv",
        "pmc_perf_7.csv",
        "pmc_perf_8.csv",
        "pmc_perf_9.csv",
        "sysinfo.csv",
        "timestamps.csv",
    ]
    if soc == "mi200":
        expected_csvs = [
            "SQ_IFETCH_LEVEL.csv",
            "SQ_INST_LEVEL_LDS.csv",
            "SQ_INST_LEVEL_SMEM.csv",
            "SQ_INST_LEVEL_VMEM.csv",
            "SQ_LEVEL_WAVES.csv",
            "pmc_perf.csv",
            "pmc_perf_0.csv",
            "pmc_perf_1.csv",
            "pmc_perf_10.csv",
            "pmc_perf_11.csv",
            "pmc_perf_12.csv",
            "pmc_perf_2.csv",
            "pmc_perf_3.csv",
            "pmc_perf_4.csv",
            "pmc_perf_5.csv",
            "pmc_perf_6.csv",
            "pmc_perf_7.csv",
            "pmc_perf_8.csv",
            "pmc_perf_9.csv",
            "roofline.csv",
            "sysinfo.csv",
            "timestamps.csv",
        ]
    assert sorted(list(file_dict.keys())) == expected_csvs

    if COUNTER_LOGGING:
        log_counter(file_dict, inspect.stack()[0][3])

    if METRIC_LOGGING:
        log_metric(
            inspect.stack()[0][3],
            {"default": {"absolute": DEFAULT_ABS_DIFF, "relative": DEFAULT_REL_DIFF}},
        )


def test_dispatch_0():
    if os.path.exists(workload_1):
        shutil.rmtree(workload_1)
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--dispatch",
                "0",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0

    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1 + "/" + file)
            if not "sysinfo" in file and not "roofline" in file:
                assert len(file_dict[file].index) == 1
    if soc == "mi200":
        
        assert sorted(list(file_dict.keys())) == ALL_CSVS_MI200
    else:
        assert sorted(list(file_dict.keys())) == ALL_CSVS

    if COUNTER_LOGGING:
        log_counter(file_dict, inspect.stack()[0][3])

    if METRIC_LOGGING:
        log_metric(
            inspect.stack()[0][3],
            {"default": {"absolute": DEFAULT_ABS_DIFF, "relative": DEFAULT_REL_DIFF}},
            [
                "--dispatch",
                "0",
            ],
        )


def test_dispatch_0_1():
    if os.path.exists(workload_1):
        shutil.rmtree(workload_1)
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--dispatch",
                "0:2",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0

    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1 + "/" + file)
            if not "sysinfo" in file and not "roofline" in file:
                assert len(file_dict[file].index) == 2

    if soc == "mi200":
        
        assert sorted(list(file_dict.keys())) == ALL_CSVS_MI200
    else:
        assert sorted(list(file_dict.keys())) == ALL_CSVS

    if COUNTER_LOGGING:
        log_counter(file_dict, inspect.stack()[0][3])

    if METRIC_LOGGING:
        log_metric(
            inspect.stack()[0][3],
            {"default": {"absolute": DEFAULT_ABS_DIFF, "relative": DEFAULT_REL_DIFF}},
            ["--dispatch", "0", "1"],
        )


def test_dispatch_2():
    if os.path.exists(workload_1):
        shutil.rmtree(workload_1)
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--dispatch",
                dispatch_id,
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0

    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1 + "/" + file)
            if not "sysinfo" in file and not "roofline" in file:
                assert len(file_dict[file].index) == 1
    if soc == "mi200":
        
        assert sorted(list(file_dict.keys())) == ALL_CSVS_MI200
    else:
        assert sorted(list(file_dict.keys())) == ALL_CSVS

    if COUNTER_LOGGING:
        log_counter(file_dict, inspect.stack()[0][3])

    if METRIC_LOGGING:
        log_metric(
            inspect.stack()[0][3],
            {"default": {"absolute": DEFAULT_ABS_DIFF, "relative": DEFAULT_REL_DIFF}},
            [
                "--dispatch",
                str(dispatch_id),
            ],
        )


def test_kernel_verbose_0():
    if os.path.exists(workload_1):
        shutil.rmtree(workload_1)
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--kernel-verbose",
                "0",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0

    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1 + "/" + file)
            if not "sysinfo" in file:
                assert len(file_dict[file].index) >= num_kernels
    if soc == "mi200":
        
        assert sorted(list(file_dict.keys())) == ALL_CSVS_MI200
    else:
        assert sorted(list(file_dict.keys())) == ALL_CSVS

    if COUNTER_LOGGING:
        log_counter(file_dict, inspect.stack()[0][3])

    if METRIC_LOGGING:
        log_metric(
            inspect.stack()[0][3],
            {"default": {"absolute": DEFAULT_ABS_DIFF, "relative": DEFAULT_REL_DIFF}},
        )


def test_kernel_verbose_1():
    if os.path.exists(workload_1):
        shutil.rmtree(workload_1)
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--kernel-verbose",
                "1",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0

    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1 + "/" + file)
            if not "sysinfo" in file:
                assert len(file_dict[file].index) >= num_kernels
    if soc == "mi200":
        
        assert sorted(list(file_dict.keys())) == ALL_CSVS_MI200
    else:
        assert sorted(list(file_dict.keys())) == ALL_CSVS

    if COUNTER_LOGGING:
        log_counter(file_dict, inspect.stack()[0][3])

    if METRIC_LOGGING:
        log_metric(
            inspect.stack()[0][3],
            {"default": {"absolute": DEFAULT_ABS_DIFF, "relative": DEFAULT_REL_DIFF}},
        )


def test_kernel_verbose_2():
    if os.path.exists(workload_1):
        shutil.rmtree(workload_1)
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--kernel-verbose",
                "2",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0

    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1 + "/" + file)
            if not "sysinfo" in file:
                assert len(file_dict[file].index) >= num_kernels
    if soc == "mi200":
        
        assert sorted(list(file_dict.keys())) == ALL_CSVS_MI200
    else:
        assert sorted(list(file_dict.keys())) == ALL_CSVS

    if COUNTER_LOGGING:
        log_counter(file_dict, inspect.stack()[0][3])

    if METRIC_LOGGING:
        log_metric(
            inspect.stack()[0][3],
            {"default": {"absolute": DEFAULT_ABS_DIFF, "relative": DEFAULT_REL_DIFF}},
        )


def test_kernel_verbose_3():
    if os.path.exists(workload_1):
        shutil.rmtree(workload_1)
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--kernel-verbose",
                "3",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0

    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1 + "/" + file)
            if not "sysinfo" in file:
                assert len(file_dict[file].index) >= num_kernels
    if soc == "mi200":
        
        assert sorted(list(file_dict.keys())) == ALL_CSVS_MI200
    else:
        assert sorted(list(file_dict.keys())) == ALL_CSVS

    if COUNTER_LOGGING:
        log_counter(file_dict, inspect.stack()[0][3])

    if METRIC_LOGGING:
        log_metric(
            inspect.stack()[0][3],
            {"default": {"absolute": DEFAULT_ABS_DIFF, "relative": DEFAULT_REL_DIFF}},
        )


def test_kernel_verbose_4():
    if os.path.exists(workload_1):
        shutil.rmtree(workload_1)
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--kernel-verbose",
                "4",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0

    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1 + "/" + file)
            if not "sysinfo" in file:
                assert len(file_dict[file].index) >= num_kernels
    if soc == "mi200":
        
        assert sorted(list(file_dict.keys())) == ALL_CSVS_MI200
    else:
        assert sorted(list(file_dict.keys())) == ALL_CSVS

    if COUNTER_LOGGING:
        log_counter(file_dict, inspect.stack()[0][3])

    if METRIC_LOGGING:
        log_metric(
            inspect.stack()[0][3],
            {"default": {"absolute": DEFAULT_ABS_DIFF, "relative": DEFAULT_REL_DIFF}},
        )


def test_kernel_verbose_5():
    if os.path.exists(workload_1):
        shutil.rmtree(workload_1)
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--kernel-verbose",
                "5",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0

    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1 + "/" + file)
            if not "sysinfo" in file:
                assert len(file_dict[file].index) >= num_kernels
    if soc == "mi200":
        
        assert sorted(list(file_dict.keys())) == ALL_CSVS_MI200
    else:
        assert sorted(list(file_dict.keys())) == ALL_CSVS

    if COUNTER_LOGGING:
        log_counter(file_dict, inspect.stack()[0][3])

    if METRIC_LOGGING:
        log_metric(
            inspect.stack()[0][3],
            {"default": {"absolute": DEFAULT_ABS_DIFF, "relative": DEFAULT_REL_DIFF}},
        )


def test_join_type_grid():
    if os.path.exists(workload_1):
        shutil.rmtree(workload_1)
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--join-type",
                "grid",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0

    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1 + "/" + file)
            if not "sysinfo" in file:
                assert len(file_dict[file].index) >= num_kernels
    if soc == "mi200":
        
        assert sorted(list(file_dict.keys())) == ALL_CSVS_MI200
    else:
        assert sorted(list(file_dict.keys())) == ALL_CSVS

    if COUNTER_LOGGING:
        log_counter(file_dict, inspect.stack()[0][3])

    if METRIC_LOGGING:
        log_metric(
            inspect.stack()[0][3],
            {"default": {"absolute": DEFAULT_ABS_DIFF, "relative": DEFAULT_REL_DIFF}},
        )


def test_join_type_kernel():
    if os.path.exists(workload_1):
        shutil.rmtree(workload_1)
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--join-type",
                "kernel",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0

    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1 + "/" + file)
            if not "sysinfo" in file:
                assert len(file_dict[file].index) >= num_kernels
    if soc == "mi200":
        
        assert sorted(list(file_dict.keys())) == ALL_CSVS_MI200
    else:
        assert sorted(list(file_dict.keys())) == ALL_CSVS

    if COUNTER_LOGGING:
        log_counter(file_dict, inspect.stack()[0][3])

    if METRIC_LOGGING:
        log_metric(
            inspect.stack()[0][3],
            {"default": {"absolute": DEFAULT_ABS_DIFF, "relative": DEFAULT_REL_DIFF}},
        )


def test_device_0():
    if os.path.exists(workload_1):
        shutil.rmtree(workload_1)
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--device",
                "0",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0

    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1 + "/" + file)
            if not "sysinfo" in file:
                if "roofline" in file:
                    assert len(file_dict[file].index)
                else:
                    assert len(file_dict[file].index) >= num_kernels
    if soc == "mi200":
        
        assert sorted(list(file_dict.keys())) == ALL_CSVS_MI200
    else:
        assert sorted(list(file_dict.keys())) == ALL_CSVS

    if COUNTER_LOGGING:
        log_counter(file_dict, inspect.stack()[0][3])

    if METRIC_LOGGING:
        log_metric(
            inspect.stack()[0][3],
            {"default": {"absolute": DEFAULT_ABS_DIFF, "relative": DEFAULT_REL_DIFF}},
        )


def test_no_roof():
    if os.path.exists(workload_1):
        shutil.rmtree(workload_1)
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--no-roof",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0

    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1 + "/" + file)
            if not "sysinfo" in file:
                assert len(file_dict[file].index) >= num_kernels
    if soc == "mi200":
        
        assert sorted(list(file_dict.keys())) == [
            "SQ_IFETCH_LEVEL.csv",
            "SQ_INST_LEVEL_LDS.csv",
            "SQ_INST_LEVEL_SMEM.csv",
            "SQ_INST_LEVEL_VMEM.csv",
            "SQ_LEVEL_WAVES.csv",
            "pmc_perf.csv",
            "pmc_perf_0.csv",
            "pmc_perf_1.csv",
            "pmc_perf_10.csv",
            "pmc_perf_11.csv",
            "pmc_perf_12.csv",
            "pmc_perf_13.csv",
            "pmc_perf_14.csv",
            "pmc_perf_15.csv",
            "pmc_perf_16.csv",
            "pmc_perf_2.csv",
            "pmc_perf_3.csv",
            "pmc_perf_4.csv",
            "pmc_perf_5.csv",
            "pmc_perf_6.csv",
            "pmc_perf_7.csv",
            "pmc_perf_8.csv",
            "pmc_perf_9.csv",
            "sysinfo.csv",
            "timestamps.csv",
        ]
    else:
        assert sorted(list(file_dict.keys())) == ALL_CSVS

    if COUNTER_LOGGING:
        log_counter(file_dict, inspect.stack()[0][3])

    if METRIC_LOGGING:
        log_metric(
            inspect.stack()[0][3],
            {"default": {"absolute": DEFAULT_ABS_DIFF, "relative": DEFAULT_REL_DIFF}},
        )


def test_sort_dispatches():
    if os.path.exists(workload_1):
        shutil.rmtree(workload_1)
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--roof-only",
                "--sort",
                "dispatches",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    if soc == "mi100":
        # assert that it did not run
        assert e.value.code >= 1
        # Do not continue testing
        return

    # assert successful run
    assert e.value.code == 0

    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1 + "/" + file)
            if not "sysinfo" in file:
                assert len(file_dict[file].index) >= num_kernels
        elif file.endswith(".pdf"):
            file_dict[file] = "pdf"
    if soc == "mi200":
        
        assert sorted(list(file_dict.keys())) == ROOF_ONLY_FILES
    else:
        assert sorted(list(file_dict.keys())) == ALL_CSVS

    if COUNTER_LOGGING:
        log_counter(file_dict, inspect.stack()[0][3])

    if METRIC_LOGGING:
        log_metric(
            inspect.stack()[0][3],
            {"default": {"absolute": DEFAULT_ABS_DIFF, "relative": DEFAULT_REL_DIFF}},
        )


def test_sort_kernels():
    if os.path.exists(workload_1):
        shutil.rmtree(workload_1)
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--roof-only",
                "--sort",
                "kernels",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()
    if soc == "mi100":
        # assert that it did not run
        assert e.value.code >= 1
        # Do not continue testing
        return

    # assert successful run
    assert e.value.code == 0

    files_in_workload = os.listdir(workload_1)

    file_dict = {}
    # Check if csvs have data
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1 + "/" + file)
            if not "sysinfo" in file:
                assert len(file_dict[file].index) >= num_kernels
        elif file.endswith(".pdf"):
            file_dict[file] = "pdf"
    if soc == "mi200":
        
        assert sorted(list(file_dict.keys())) == ROOF_ONLY_FILES
    else:
        assert sorted(list(file_dict.keys())) == ALL_CSVS

    if COUNTER_LOGGING:
        log_counter(file_dict, inspect.stack()[0][3])

    if METRIC_LOGGING:
        log_metric(
            inspect.stack()[0][3],
            {"default": {"absolute": DEFAULT_ABS_DIFF, "relative": DEFAULT_REL_DIFF}},
        )


def test_mem_levels_HBM():
    if os.path.exists(workload_1):
        shutil.rmtree(workload_1)
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--roof-only",
                "--mem-level",
                "HBM",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    if soc == "mi100":
        # assert that it did not run
        assert e.value.code >= 1
        # Do not continue testing
        return

    # assert successful run
    assert e.value.code == 0

    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1 + "/" + file)
            if not "sysinfo" in file:
                assert len(file_dict[file].index) >= num_kernels
        elif file.endswith(".pdf"):
            file_dict[file] = "pdf"
    if soc == "mi200":
        
        assert sorted(list(file_dict.keys())) == ROOF_ONLY_FILES
    else:
        assert sorted(list(file_dict.keys())) == ALL_CSVS

    if COUNTER_LOGGING:
        log_counter(file_dict, inspect.stack()[0][3])

    if METRIC_LOGGING:
        log_metric(
            inspect.stack()[0][3],
            {"default": {"absolute": DEFAULT_ABS_DIFF, "relative": DEFAULT_REL_DIFF}},
        )


def test_mem_levels_L2():
    if os.path.exists(workload_1):
        shutil.rmtree(workload_1)
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--roof-only",
                "--mem-level",
                "L2",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    if soc == "mi100":
        # assert that it did not run
        assert e.value.code >= 1
        # Do not continue testing
        return

    # assert successful run
    assert e.value.code == 0

    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1 + "/" + file)
            if not "sysinfo" in file:
                assert len(file_dict[file].index) >= num_kernels
        elif file.endswith(".pdf"):
            file_dict[file] = "pdf"
    if soc == "mi200":
        
        assert sorted(list(file_dict.keys())) == ROOF_ONLY_FILES
    else:
        assert sorted(list(file_dict.keys())) == ALL_CSVS

    if COUNTER_LOGGING:
        log_counter(file_dict, inspect.stack()[0][3])

    if METRIC_LOGGING:
        log_metric(
            inspect.stack()[0][3],
            {"default": {"absolute": DEFAULT_ABS_DIFF, "relative": DEFAULT_REL_DIFF}},
        )


def test_mem_levels_vL1D():
    if os.path.exists(workload_1):
        shutil.rmtree(workload_1)
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--roof-only",
                "--mem-level",
                "vL1D",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()
    if soc == "mi100":
        # assert that it did not run
        assert e.value.code >= 1
        # Do not continue testing
        return

    # assert successful run
    assert e.value.code == 0

    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1 + "/" + file)
            if not "sysinfo" in file:
                assert len(file_dict[file].index) >= num_kernels
        elif file.endswith(".pdf"):
            file_dict[file] = "pdf"
    if soc == "mi200":
        
        assert sorted(list(file_dict.keys())) == ROOF_ONLY_FILES
    else:
        assert sorted(list(file_dict.keys())) == ALL_CSVS

    if COUNTER_LOGGING:
        log_counter(file_dict, inspect.stack()[0][3])

    if METRIC_LOGGING:
        log_metric(
            inspect.stack()[0][3],
            {"default": {"absolute": DEFAULT_ABS_DIFF, "relative": DEFAULT_REL_DIFF}},
        )


def test_mem_levels_LDS():
    if os.path.exists(workload_1):
        shutil.rmtree(workload_1)
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--roof-only",
                "--mem-level",
                "LDS",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()
    if soc == "mi100":
        # assert that it did not run
        assert e.value.code >= 1
        # Do not continue testing
        return

    # assert successful run
    assert e.value.code == 0

    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1 + "/" + file)
            if not "sysinfo" in file:
                assert len(file_dict[file].index) >= num_kernels
        elif file.endswith(".pdf"):
            file_dict[file] = "pdf"
    if soc == "mi200":
        
        assert sorted(list(file_dict.keys())) == ROOF_ONLY_FILES
    else:
        assert sorted(list(file_dict.keys())) == ALL_CSVS

    if COUNTER_LOGGING:
        log_counter(file_dict, inspect.stack()[0][3])

    if METRIC_LOGGING:
        log_metric(
            inspect.stack()[0][3],
            {"default": {"absolute": DEFAULT_ABS_DIFF, "relative": DEFAULT_REL_DIFF}},
        )


def test_mem_levels_HBM_LDS():
    if os.path.exists(workload_1):
        shutil.rmtree(workload_1)
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--roof-only",
                "--mem-level",
                "HBM",
                "LDS",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()
    if soc == "mi100":
        # assert that it did not run
        assert e.value.code >= 1
        # Do not continue testing
        return

    # assert successful run
    assert e.value.code == 0

    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1 + "/" + file)
            if not "sysinfo" in file:
                assert len(file_dict[file].index) >= num_kernels
        elif file.endswith(".pdf"):
            file_dict[file] = "pdf"
    if soc == "mi200":
        
        assert sorted(list(file_dict.keys())) == ROOF_ONLY_FILES
    else:
        assert sorted(list(file_dict.keys())) == ALL_CSVS

    if COUNTER_LOGGING:
        log_counter(file_dict, inspect.stack()[0][3])

    if METRIC_LOGGING:
        log_metric(
            inspect.stack()[0][3],
            {"default": {"absolute": DEFAULT_ABS_DIFF, "relative": DEFAULT_REL_DIFF}},
        )


def test_mem_levels_vL1D_LDS():
    if os.path.exists(workload_1):
        shutil.rmtree(workload_1)
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--roof-only",
                "--mem-level",
                "vL1D",
                "LDS",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()
    if soc == "mi100":
        # assert that it did not run
        assert e.value.code >= 1
        # Do not continue testing
        return

    # assert successful run
    assert e.value.code == 0

    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1 + "/" + file)
            if not "sysinfo" in file:
                assert len(file_dict[file].index) >= num_kernels
        elif file.endswith(".pdf"):
            file_dict[file] = "pdf"
    if soc == "mi200":
        
        assert sorted(list(file_dict.keys())) == ROOF_ONLY_FILES
    else:
        assert sorted(list(file_dict.keys())) == ALL_CSVS

    if COUNTER_LOGGING:
        log_counter(file_dict, inspect.stack()[0][3])

    if METRIC_LOGGING:
        log_metric(
            inspect.stack()[0][3],
            {"default": {"absolute": DEFAULT_ABS_DIFF, "relative": DEFAULT_REL_DIFF}},
        )


def test_mem_levels_L2_vL1D_LDS():
    if os.path.exists(workload_1):
        shutil.rmtree(workload_1)
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--roof-only",
                "--mem-level",
                "L2",
                "vL1D",
                "LDS",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()
    if soc == "mi100":
        # assert that it did not run
        assert e.value.code >= 1
        # Do not continue testing
        return
    # assert successful run
    assert e.value.code == 0

    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1 + "/" + file)
            if not "sysinfo" in file:
                assert len(file_dict[file].index) >= num_kernels
        elif file.endswith(".pdf"):
            file_dict[file] = "pdf"
    if soc == "mi200":
        
        assert sorted(list(file_dict.keys())) == ROOF_ONLY_FILES
    else:
        assert sorted(list(file_dict.keys())) == ALL_CSVS

    if COUNTER_LOGGING:
        log_counter(file_dict, inspect.stack()[0][3])

    if METRIC_LOGGING:
        log_metric(
            inspect.stack()[0][3],
            {"default": {"absolute": DEFAULT_ABS_DIFF, "relative": DEFAULT_REL_DIFF}},
        )


def test_kernel_names():
    if os.path.exists(workload_1):
        shutil.rmtree(workload_1)
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--roof-only",
                "--kernel-names",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    if soc == "mi100":
        # assert that it did not run
        assert e.value.code >= 1
        # Do not continue testing
        return
    # assert successful run
    assert e.value.code == 0

    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1 + "/" + file)
            if not "sysinfo" in file:
                assert len(file_dict[file].index) >= num_kernels
        elif file.endswith(".pdf"):
            file_dict[file] = "pdf"
    if soc == "mi200":
        
        assert sorted(list(file_dict.keys())) == [
            "empirRoof_gpu-ALL_fp32.pdf",
            "empirRoof_gpu-ALL_int8_fp16.pdf",
            "kernelName_legend.pdf",
            "pmc_perf.csv",
            "pmc_perf_0.csv",
            "pmc_perf_1.csv",
            "pmc_perf_2.csv",
            "roofline.csv",
            "sysinfo.csv",
            "timestamps.csv",
        ]
    else:
        assert sorted(list(file_dict.keys())) == ALL_CSVS

    if COUNTER_LOGGING:
        log_counter(file_dict, inspect.stack()[0][3])

    if METRIC_LOGGING:
        log_metric(
            inspect.stack()[0][3],
            {"default": {"absolute": DEFAULT_ABS_DIFF, "relative": DEFAULT_REL_DIFF}},
        )
