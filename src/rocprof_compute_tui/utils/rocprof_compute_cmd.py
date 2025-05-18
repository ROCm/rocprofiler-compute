import logging
import shlex
import subprocess
from pathlib import Path
from typing import List, Optional, Union


class RocprofRunner:
    """
    Encapsulates calls to the 'rocprof-compute' CLI.
    """

    # FIXME: should use executable "rocprof-compute"
    def __init__(self, executable: str = "rocprof-compute"):
        self.executable = executable
        self.logger = logging.getLogger(__name__)

    def run_profile(
        self,
        target_app: str,
        app_args: Optional[List[str]] = None,
        output_dir: Union[str, Path] = "rocprof_output",
        real_time_logs: bool = False,
        timeout: Optional[int] = None,
    ) -> int:

        cmd_parts = [self.executable, "profile"]
        # Set output directory
        cmd_parts += ["--output-dir", str(output_dir)]

        # The actual app to profile + its args
        cmd_parts += ["--", target_app]
        if app_args:
            cmd_parts += app_args

        cmd_str = " ".join(shlex.quote(part) for part in cmd_parts)
        self.logger.info("Running profiling command: %s", cmd_str)

        # Now run the subprocess
        process = subprocess.Popen(
            cmd_parts, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        if real_time_logs:
            # Stream the output to logs or console while the process runs
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    self.logger.debug("[rocprof-compute stdout] %s", line.rstrip())
            # In real-time mode, also read remaining stderr if you want
            stderr_output = process.stderr.read()
            if stderr_output:
                self.logger.debug("[rocprof-compute stderr] %s", stderr_output.rstrip())
        else:
            # Non-streaming: just wait for it to complete, capturing output
            stdout_output, stderr_output = process.communicate(timeout=timeout)
            if stdout_output:
                self.logger.debug("[rocprof-compute stdout] %s", stdout_output)
            if stderr_output:
                self.logger.debug("[rocprof-compute stderr] %s", stderr_output)

        exit_code = process.returncode
        self.logger.info("rocprof-compute exited with code %s", exit_code)

        return exit_code

    def run_analyze(
        self,
        input_dir: Union[str, Path],
        output_file: Optional[Union[str, Path]] = None,
        extra_args: Optional[List[str]] = None,
    ):

        cmd_parts = [self.executable, "analyze", "--path", str(input_dir)]
        if output_file:
            cmd_parts += ["--output", str(output_file)]
        if extra_args:
            cmd_parts += extra_args

        cmd_str = " ".join(shlex.quote(part) for part in cmd_parts)
        self.logger.info("Running analyze command: %s", cmd_str)

        process = subprocess.Popen(
            cmd_parts, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        stdout_output, stderr_output = process.communicate()
        exit_code = process.returncode

        return stdout_output, stderr_output, exit_code, cmd_str
