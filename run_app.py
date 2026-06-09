import os
import signal
import socket
import subprocess
import time

from web_app import HOST, PORT, main


def find_server_pids():
    result = subprocess.run(
        ["lsof", f"-tiTCP:{PORT}", "-sTCP:LISTEN"],
        capture_output=True,
        text=True,
        check=False,
    )
    return [
        int(pid)
        for pid in result.stdout.splitlines()
        if pid.strip().isdigit() and int(pid) != os.getpid()
    ]


def port_is_busy():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.1)
        return sock.connect_ex((HOST, PORT)) == 0


def stop_existing_server():
    pids = find_server_pids()
    for pid in pids:
        os.kill(pid, signal.SIGTERM)

    deadline = time.time() + 5
    while pids and port_is_busy() and time.time() < deadline:
        time.sleep(0.1)


if __name__ == "__main__":
    stop_existing_server()
    main()
