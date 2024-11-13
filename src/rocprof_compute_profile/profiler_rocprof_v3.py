##############################################################################bl
# MIT License
#
# Copyright (c) 2021 - 2024 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
##############################################################################el

import os
import shlex
from rocprof_compute_profile.profiler_base import RocProfCompute_Base
from utils.utils import (
    demarcate,
    console_log,
    console_error,
    replace_timestamps,
)


class rocprof_v3_profiler(RocProfCompute_Base):
    def __init__(self, profiling_args, profiler_mode, soc):
        super().__init__(profiling_args, profiler_mode, soc)
        self.ready_to_profile = (
            self.get_args().roof_only
            and not os.path.isfile(os.path.join(self.get_args().path, "pmc_perf.csv"))
            or not self.get_args().roof_only
        )

    def get_profiler_options(self, fname):
        app_cmd = shlex.split(self.get_args().remaining)
        output_format = "csv"
        if "ROCPROF_OUTPUT_FORMAT" in os.environ.keys():
            output_format = os.environ["ROCPROF_OUTPUT_FORMAT"].lower()

        if output_format not in ["csv", "json"]:
            console_error("Invalid rocprof output format", True)

        args = [
            # v3 requires output directory argument
            "-d",
            self.get_args().path + "/" + "out",
            "--kernel-trace",
            "--output-format",
            output_format,
            "--"
        ]
        args.extend(app_cmd)
        return args

    # -----------------------
    # Required child methods
    # -----------------------
    @demarcate
    def pre_processing(self):
        """Perform any pre-processing steps prior to profiling."""
        super().pre_processing()

    @demarcate
    def run_profiling(self, version, prog):
        """Run profiling."""
        if self.ready_to_profile:
            if self.get_args().roof_only:
                console_log(
                    "roofline", "Generating pmc_perf.csv (roofline counters only)."
                )
            # Log profiling options and setup filtering
            super().run_profiling(version, prog)
        else:
            console_log("roofline", "Detected existing pmc_perf.csv")

    @demarcate
    def post_processing(self):
        """Perform any post-processing steps prior to profiling."""
        super().post_processing()

        if self.ready_to_profile:
            # Manually join each pmc_perf*.csv output
            self.join_prof()
            # Replace timestamp data to solve a known rocprof bug
            #replace_timestamps(self.get_args().path)
