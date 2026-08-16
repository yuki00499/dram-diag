"""停止后台运行的标注站服务（--detach 模式启动的进程）。"""

import subprocess
import sys


def main():
    if sys.platform != "win32":
        print("仅支持 Windows。")
        return
    result = subprocess.run(
        ["wmic", "process", "where", "name='python.exe'", "get", "ProcessId,CommandLine", "/format:list"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    targets = []
    lines = result.stdout.splitlines()
    current = ""
    for line in lines:
        if line.startswith("CommandLine="):
            current = line[len("CommandLine="):]
        elif line.startswith("ProcessId="):
            pid = line[len("ProcessId="):].strip()
            if pid and "build_multi_label_app.py --serve" in current:
                targets.append(int(pid))
            current = ""
    if not targets:
        print("没有找到后台标注站进程。")
        return
    for pid in targets:
        try:
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], check=True)
            print(f"已停止标注站进程 PID {pid}")
        except subprocess.CalledProcessError:
            print(f"停止 PID {pid} 失败（可能已退出）")


if __name__ == "__main__":
    main()
