import streamlit as st
import subprocess
import sys
import os

# This is a simple wrapper to run the direct_test.py file
if __name__ == "__main__":
    # Get the absolute path to the direct_test.py file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_script = os.path.join(current_dir, "direct_test.py")
    
    # Run the streamlit app
    print(f"Running test script: {test_script}")
    subprocess.run([sys.executable, "-m", "streamlit", "run", test_script]) 