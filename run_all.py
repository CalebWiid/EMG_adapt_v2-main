import subprocess, sys

scripts = ["Plot_cca.py", "cluade_reptile.py"]   # EXACT names/case

for s in scripts:
    print(f"\n{'='*60}\n Running {s}\n{'='*60}", flush=True)
    result = subprocess.run([sys.executable, s])
    if result.returncode != 0:
        print(f"!!! {s} failed (exit {result.returncode}) — stopping.")
        sys.exit(result.returncode)
print("\nAll scripts completed.")