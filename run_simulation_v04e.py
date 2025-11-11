# run_simulation_v04e.py
# Интерактивное меню для ACE Engine v0.4e-fix
# Запускает main.py, xi_chart_v04e.py, auto_tune_v04e.py и просмотр отчёта.

import os
import subprocess

BANNER = """
=====================================================
 ACE Engine v0.4e-fix — Simulation on Runner
=====================================================
Choose an option:
  1. Run main simulation (generate report and charts)
  2. Generate Xi charts (advanced visualization)
  3. Auto-tune parameters (optimization)
  4. View last report
  5. Exit
=====================================================
"""

def run(cmd):
    """Запускает внешний Python-скрипт и ждёт завершения."""
    try:
        subprocess.run(["python", cmd], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[!] Error while running {cmd}: {e}")

def view_last_report():
    path = "last_report.txt"
    if not os.path.exists(path):
        print("No report found. Run simulation first.")
        return
    with open(path, "r", encoding="utf-8") as f:
        print(f.read())

def main():
    while True:
        print(BANNER)
        choice = input("Enter your choice (1-5): ").strip()
        if choice == "1":
            run("main.py")
        elif choice == "2":
            run("xi_chart_v04e.py")
        elif choice == "3":
            run("auto_tune_v04e.py")
        elif choice == "4":
            view_last_report()
        elif choice == "5":
            print("Exiting ACE Simulation Runner.")
            break
        else:
            print("Invalid choice. Please enter 1–5.")
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
