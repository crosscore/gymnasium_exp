import subprocess
import sys
import os

# List of scripts to run in order
scripts_to_run = [
    "q.py",
    "sarsa.py",
    "expected_sarsa.py",
    "double_q.py",
]

# Get the path to the python interpreter currently running this script
python_executable = sys.executable

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

def run_script(script_name):
    """Runs a given python script using the same interpreter."""
    script_path = os.path.join(script_dir, script_name)
    if not os.path.exists(script_path):
        print(f"Error: Script not found - {script_path}")
        return False
    
    print(f"--- Running {script_name} ---")
    try:
        # Run the script and capture output
        result = subprocess.run(
            [python_executable, script_path],
            check=True, # Raise exception if script returns non-zero exit code
            capture_output=True, # Capture stdout and stderr
            text=True, # Decode stdout/stderr as text
            cwd=script_dir # Ensure script runs in its own directory context
        )
        print(f"--- Finished {script_name} ---")
        # print("Output:") # Optionally print the script's output
        # print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"!!! Error running {script_name} !!!")
        print(f"Return code: {e.returncode}")
        print("Stderr:")
        print(e.stderr)
        return False
    except Exception as e:
        print(f"!!! An unexpected error occurred running {script_name}: {e} !!!")
        return False

if __name__ == "__main__":
    all_successful = True
    for script in scripts_to_run:
        success = run_script(script)
        if not success:
            all_successful = False
            print(f"Stopping execution due to error in {script}.")
            break # Stop if one script fails
    
    if all_successful:
        print("\n=== All scripts executed successfully! ===")
    else:
        print("\n=== Execution finished with errors. ===") 