import logging
import os
import subprocess
import sys
import time
from io import StringIO, BytesIO
from multiprocessing import Semaphore
from shutil import which
from threading import Lock
from typing import Union

logger = logging.getLogger(__name__)


class BaseProcInvoker(object):

    def __init__(self, command_basename: str, version=None):
        self.command_basename = command_basename
        self.version = version
        _all_invokers.append(self)

    def run(self, arguments: list[str], **kwargs) -> int:
        raise NotImplemented()

    def check_output(self, arguments: list[str], **kwargs) -> str:
        raise NotImplemented()

    def Popen(self, arguments: list[str], **kwargs) -> subprocess.Popen[str]:
        raise NotImplemented()

    def present(self) -> bool:
        """
        Check if the command is present. If it is possible to install it, that will happen. This is normally used by
        optional tools, rather than attempting to execute and caching the failure.
        :return: True if the command is present and may be used.
        """
        return False

    def required(self) -> bool:
        """
        Check if this tool is required.
        :return: True if this tool is required, False if processing can continue in other ways.
        """
        return True

    def install(self) -> bool:
        """
        Request the tool to be installed.
        :return: True if the tool was successfully installed.
        """
        return False

    def array_as_command(self, arguments: list[str]) -> str:
        command = [self.command_basename]
        for e in arguments:
            if '\'' in e:
                command.append('"' + e + '"')
            else:
                command.append("'" + e + "'")
        return ' '.join(command)


class SubprocessProcInvoker(BaseProcInvoker):
    # TODO: add method to install or prompt user for install instructions

    def __init__(self, command_basename: str, version_parser=None, version_target: list[str] = None, required=True,
                 semaphore: Union[None, Semaphore] = None):
        super().__init__(command_basename, version=None)
        self._once_lock = Lock()
        self.command_path = None
        self.version_parser = version_parser
        self.version_target = version_target
        self._required = required
        self.semaphore = semaphore

    def _run(self, arguments: list[str], kwargs) -> int:
        log_output = False
        if kwargs.get('capture_output') is None and kwargs.get('stdout') is None and kwargs.get('stderr') is None:
            log_output = True
            kwargs['capture_output'] = True
            kwargs['text'] = False

        result = subprocess.run(arguments, **kwargs)

        if log_output:
            if result.stdout:
                try:
                    logger.debug(str(result.stdout))
                except:
                    pass
            if result.stderr:
                try:
                    logger.debug(str(result.stderr))
                except:
                    pass

        if result.returncode == 130:
            raise KeyboardInterrupt()

        return result.returncode

    def run(self, arguments: list[str], **kwargs) -> int:
        if self.semaphore:
            self.semaphore.acquire()

        try:
            result = self._run(self._build_command(arguments), kwargs)
        finally:
            if self.semaphore:
                self.semaphore.release()

        return result

    def _check_output(self, arguments: list[str], kwargs) -> str:
        return subprocess.check_output(arguments, **kwargs)

    def check_output(self, arguments: list[str], **kwargs) -> str:
        if self.semaphore:
            self.semaphore.acquire()

        try:
            result = self._check_output(self._build_command(arguments), kwargs)
        finally:
            if self.semaphore:
                self.semaphore.release()

        return result

    def _Popen(self, arguments: list[str], kwargs) -> subprocess.Popen[str]:
        return subprocess.Popen(arguments, **kwargs)

    def Popen(self, arguments: list[str], **kwargs) -> subprocess.Popen[str]:
        result = self._Popen(self._build_command(arguments), kwargs)
        return result

    def _build_command(self, arguments: list[str]) -> list[str]:
        command_array = [self._get_command()]
        if arguments:
            command_array.extend(arguments)
        logger.debug(self.array_as_command(command_array))
        return command_array

    def _get_command(self):
        if self.command_path is None:
            self._once_lock.acquire()
            try:
                if self.command_path is None:
                    _maybe_path = self._find_or_install()
                    if self.version_parser is not None:
                        self.version = self.version_parser(_maybe_path)
                    self.command_path = _maybe_path
            finally:
                self._once_lock.release()
        return self.command_path

    def _find_or_install(self) -> Union[None, str]:
        command = self.find_command()
        if command is None or not os.path.exists(command):
            if self.install():
                command = self.find_command()

        if command is None or not os.path.exists(command):
            if self.required():
                raise FileNotFoundError(f"{self.command_basename} not found")
            else:
                logger.info(f"{self.command_basename} not found and not required")

        return command

    def find_command(self) -> Union[None, str]:
        return which(self.command_basename)

    def required(self) -> bool:
        return self._required

    def install(self) -> bool:
        logger.error('Install "%s" %s and try again.', self.command_basename, ' '.join(self.version_target or []))
        return False

    def present(self) -> bool:
        command_path = self._get_command()
        return command_path is not None and os.access(command_path, os.X_OK)

    def _is_filename(self, filename: str) -> bool:
        if not filename:
            return False
        if os.path.exists(filename):
            return True
        has_ext = len(os.path.splitext(filename)[1]) > 0
        return has_ext

    def _find_filename_in_arguments(self, args: list[str]) -> Union[str, None]:
        primary_filename = None
        secondary_filename = None
        for idx, arg in enumerate(args[1:]):
            if arg.startswith('-'):
                if '=' in arg:
                    value = arg.split('=')[1]
                    if self._is_filename(value):
                        secondary_filename = value
            elif '.mkv' in arg:
                primary_filename = arg
            elif self._is_filename(arg):
                primary_filename = primary_filename or arg
        result = primary_filename or secondary_filename
        if result:
            return os.path.basename(result)
        return result


class MockProcInvoker(BaseProcInvoker):

    def __init__(self, command_basename: str, version=None, mocks=None, required: bool = True):
        super().__init__(command_basename, version)
        if mocks is None:
            mocks = []
        self.mocks = mocks
        self.calls = []
        self._present = len(mocks) > 0
        self._required = required

    def run(self, arguments: list[str], **kwargs) -> int:
        self.calls.append({'method': 'run', 'arguments': arguments, 'kwargs': kwargs})
        return self._get_mock('run', arguments, **kwargs)

    def check_output(self, arguments: list[str], **kwargs) -> str:
        self.calls.append({'method': 'check_output', 'arguments': arguments, 'kwargs': kwargs})
        return self._get_mock('check_output', arguments, **kwargs)

    def Popen(self, arguments: list[str], **kwargs) -> subprocess.Popen[str]:
        self.calls.append({'method': 'Popen', 'arguments': arguments, 'kwargs': kwargs})
        result = self._get_mock('Popen', arguments, **kwargs)
        if isinstance(result, subprocess.Popen):
            return result
        result_popen = subprocess.Popen(['true'])
        if isinstance(result, str):
            result_popen.stdout = StringIO(str(result))
        elif isinstance(result, bytes):
            result_popen.stdout = BytesIO(result)
        return result_popen

    def present(self) -> bool:
        return self._present

    def required(self) -> bool:
        return self._required

    def verify(self):
        """
        Verify all mocks were consumed.
        :return:
        """
        if len(self.mocks) > 0:
            raise AssertionError(f"{len(self.mocks)} mocks remaining: {self.mocks}")

    def _get_mock(self, method_name: str, arguments: list[str], **kwargs):
        if len(self.mocks) == 0:
            raise ChildProcessError(
                f"missing mock for subprocess.{method_name}({[self.command_basename] + arguments}, {kwargs})")
        mock = self.mocks[0]
        del self.mocks[0]
        if callable(mock):
            return mock(method_name, arguments, **kwargs)
        if isinstance(mock, dict) and 'result' in mock:
            if 'method_name' in mock:
                if mock['method_name'] != method_name:
                    raise ChildProcessError(f"unexpected method: {method_name} != {mock['method_name']}")
            if 'arguments' in mock:
                if mock['arguments'] != arguments:
                    raise ChildProcessError(f"unexpected arguments: {arguments} != {mock['arguments']}")
            if 'kwargs' in mock:
                if mock['kwargs'] != kwargs:
                    raise ChildProcessError(f"unexpected kwargs: {kwargs} != {mock['kwargs']}")
            return mock['result']
        return mock


_all_invokers: list[BaseProcInvoker] = []


def pre_flight_check():
    """
    Ensure all required invokers are available. Exit the process if not.
    :return:
    """
    logger.info("verifying required tools")
    failed = False
    for invoker in _all_invokers:
        if invoker.required():
            try:
                invoker.present()
                logger.info("%s: found", invoker.command_basename)
            except FileNotFoundError:
                failed = True
                logger.fatal("%s: not found", invoker.command_basename)
    if failed:
        sys.exit(255)


class StreamCapture(object):
    def __init__(self, name: str, logger=None, level=logging.INFO):
        self.name = name
        self.logger = logger
        self.level = level
        self.captured = []
        self.save = getattr(sys, name)
        setattr(sys, name, self)

    def __getstate__(self):
        return self.name, self.logger.name, self.level, self.captured

    def __setstate__(self, state):
        if self.save:
            setattr(sys, self.name, self.save)
        self.name = state[0]
        self.save = getattr(sys, self.name)
        setattr(sys, self.name, self)
        self.logger = logging.getLogger(state[1])
        self.level = state[2]
        self.captured = state[3]

    def isatty(self) -> bool:
        if self.save and hasattr(self.save, 'isatty'):
            return self.save.isatty()
        return False

    def write(self, data):
        if self.logger:
            if data.endswith("\n"):
                self.captured.append(data[:-1])
                self.logger.log(self.level, ''.join(self.captured))
                self.captured.clear()
            else:
                self.captured.append(data)
        else:
            self.captured.append(data)

    def flush(self):
        if self.logger and self.captured:
            self.logger.log(self.level, ''.join(self.captured))
            self.captured.clear()

    def finish(self, output=True):
        setattr(sys, self.name, self.save)
        if self.logger is None and output:
            target = getattr(sys, self.name)
            for line in self.captured:
                target.write(line)
            target.flush()


class ProcessStreamGenerator(object):

    def __init__(self, proc: subprocess.Popen):
        self.proc = proc
        self.stdout = []
        self.stderr = []

    def stdout_str(self) -> str:
        return ''.join(self.stdout)

    def stderr_str(self) -> str:
        return ''.join(self.stderr)

    def generator(self) -> tuple[Union[str, None], Union[str, None]]:
        is_stdout = self.proc.stdout is not None
        is_stderr = self.proc.stderr is not None
        if not is_stdout and not is_stderr:
            return
        if is_stdout:
            os.set_blocking(self.proc.stdout.fileno(), False)
        if is_stderr:
            os.set_blocking(self.proc.stderr.fileno(), False)
        while self.proc.poll() is None:
            if is_stderr:
                line_err = self.proc.stderr.readline()
                if line_err:
                    self.stderr.append(line_err)
            else:
                line_err = None

            if is_stdout:
                line_out = self.proc.stdout.readline()
                if line_out:
                    self.stdout.append(line_out)
            else:
                line_out = None

            if not line_err and not line_out:
                time.sleep(0.2)
            else:
                yield line_out, line_err
