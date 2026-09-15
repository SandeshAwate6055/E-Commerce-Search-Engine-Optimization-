"""
Bulletproof Windows Background Process Launcher (True Windows nohup).
Uses kernel-level DETACHED_PROCESS flag so closing terminal never kills the job.
"""
import os
import sys
import subprocess
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

def main():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    python_exe = os.path.join(root, ".venv_gpu", "Scripts", "python.exe")
    log_out_path = os.path.join(root, "indexing.log")
    log_err_path = os.path.join(root, "indexing_error.log")

    print("=" * 60)
    print("Launching Offline Indexing in True Background (NOHUP)")
    print(f"Working Directory: {root}")
    print(f"Stdout Log:        {log_out_path}")
    print(f"Stderr Log:        {log_err_path}")
    print("=" * 60)

    # Open file handles for direct OS-level writing
    out_file = open(log_out_path, "a", encoding="utf-8", buffering=1)
    err_file = open(log_err_path, "a", encoding="utf-8", buffering=1)

    # Windows kernel flags:
    # 0x00000008 = DETACHED_PROCESS (Process has no console; survives terminal close)
    # 0x00000200 = CREATE_NEW_PROCESS_GROUP
    DETACHED_FLAGS = 0x00000008 | 0x00000200

    cmd = [python_exe, "-u", "-m", "backend.indexing.run_indexing"]
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])

    proc = subprocess.Popen(
        cmd,
        cwd=root,
        stdout=out_file,
        stderr=err_file,
        creationflags=DETACHED_FLAGS,
        close_fds=True
    )

    time.sleep(1)

    print(f"[OK] Background process started successfully!")
    print(f"[OK] Process ID (PID): {proc.pid}")
    print("=" * 60)
    print("You can safely close this terminal window at any time!")
    print("To watch live progress, run:")
    print("   Get-Content -Wait -Tail 20 indexing.log")
    print("=" * 60)

if __name__ == "__main__":
    main()
