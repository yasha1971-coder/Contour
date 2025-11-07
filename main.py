# Orchestrator for ACE v0.4d
import subprocess, sys

def run(cmd):
    print(">", " ".join(cmd)); sys.stdout.flush()
    subprocess.check_call(cmd)

if __name__ == "__main__":
    run([sys.executable, "ace_kernel_v04d.py"])
    run([sys.executable, "xi_chart_v04d.py"])
    print("Done. See: ace_log_v04d.csv and ace_v04d_report/")
