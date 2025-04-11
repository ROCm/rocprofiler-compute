import subprocess
import shlex
import logging
import os
from pathlib import Path
from typing import List, Optional, Union


class RocprofRunner:
    """
    Encapsulates calls to the 'rocprof-compute' CLI tool,
    including building command lines, running profiling,
    capturing output, and optionally streaming logs for real-time feedback.
    """

    def __init__(self, executable: str = "src/rocprof-compute"):
        """
        :param executable: Path or name of the rocprof-compute CLI executable.
        """
        self.executable = executable
        self.logger = logging.getLogger(__name__)  # or a dedicated logger

    def run_profile(
        self,
        target_app: str,
        app_args: Optional[List[str]] = None,
        output_dir: Union[str, Path] = "rocprof_output",
        real_time_logs: bool = False,
        timeout: Optional[int] = None,
    ) -> int:
        """
        Invokes the 'rocprof-compute' profiler on the given target application.

        :param target_app: Path to the target application executable or script.
        :param app_args: List of arguments for the target app.
        :param output_dir: Directory where rocprof-compute would store results.
        :param real_time_logs: If True, print or capture stdout/stderr in real-time.
        :param timeout: If set, kill the process after N seconds.
        :return: The exit code of the profiling process (0 means success, typically).
        """

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
    ) -> int:
        """
        Runs the 'rocprof-compute analyze' command on a specified directory
        containing profiler data, optionally saving analysis to a file.

        :param input_dir: Directory where profiling data is located.
        :param output_file: Optionally specify a file to write analysis results.
        :param extra_args: Additional flags or arguments for the analyze subcommand.
        :return: The exit code of the analyze process.
        """

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

        if stdout_output:
            self.logger.debug("[rocprof-compute analyze stdout] %s", stdout_output)
        if stderr_output:
            self.logger.debug("[rocprof-compute analyze stderr] %s", stderr_output)

        self.logger.info("rocprof-compute analyze exited with code %s", exit_code)
        return exit_code
