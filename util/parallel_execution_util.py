import concurrent.futures
import subprocess
import sys

from . import terminal_interaction_util
import pprint


def _worker(command_str, process_number):
    assert isinstance(command_str, str)
    print(f"{__name__} MESSAGE: Process {process_number} started...")
    # TODO: shell=Trueとしているけど、もしかしたらこの実装は自分のshellの環境に依存したような書き方になっているかもしれない。他のComputerでも動くのか?
    try:
        p = subprocess.Popen(command_str, shell=True)
    except Exception as e:
        print(e)
        terminal_interaction_util.terminal_message_start_print()
        pprint.pprint(f"{command_str=}")
        terminal_interaction_util.terminal_message_end_print()
        # TODO: How to check if a string is a valid shell command using Python? とか調べる。
        print("Check terminal command printed above!! Probably, this command is somehow not executable.....多分Windowsでは'python3'コマンドではなく、'python'コマンドを使うとか、ある。いろいろコンピュータの環境に依存している実装になっているので注意。")
        sys.exit()

    p.wait()
    print(f"{__name__} MESSAGE: Process {process_number} finished...")

    return None


def execute_list_of_commands_parallely(list_of_shell_command_strings, n_of_parallel_processes):
    # TODO: Security Problem - Anybody can execute malicious code here just by replacing this variable!! So be careful!
    #   Quote from https://docs.python.org/3/library/subprocess.html:
    #       Unlike some other popen functions, this implementation will never implicitly call a system shell. This means that all characters, including shell metacharacters, can safely be passed to child processes.
    #       If the shell is invoked explicitly, via shell=True, it is the application’s responsibility to ensure that all whitespace and metacharacters are quoted appropriately to avoid shell injection vulnerabilities. On some platforms, it is possible to use shlex.quote() for this escaping.

    tasks = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_of_parallel_processes) as pool:
        for i in range(len(list_of_shell_command_strings)):
            tasks.append(pool.submit(
                _worker, list_of_shell_command_strings[i], i))

    print(f"{__name__} MESSAGE: Waiting for tasks...", flush=True)
    for task in concurrent.futures.as_completed(tasks):
        print(f"{__name__} MESSAGE: the result of the function '{_worker.__name__}' is: {task.result()}", flush=True)

    print(f"{__name__} MESSAGE: All processes done "
          f"(not necessarily meaning that all processes succeeded. Even if an error happens in one process, next process is executed.)")

    '''
    procs = []
    for a_command in list_of_shell_command_strings:
        a_proc = subprocess.Popen(a_command, shell=True)
        procs.append(a_proc)
    for proc in procs:
        out, err = proc.communicate()
        print(out.decode('utf-8'))
        if err is not None:
            print(err.decode('utf-8'))
    '''
