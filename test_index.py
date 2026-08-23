# test_index.py
import os
import sys

def run_all_tests():
    """
    Executes the full test suite and workflow simulation for universal_dental_ai.
    Verifies image preprocessing, data schema, and report rendering pipelines.
    """
    print("="*60)
    print("UNIVERSAL DENTAL AI - END-TO-END PIPELINE SYSTEM INTEGRATION TEST")
    print("="*60)
    
    # 1. Environment & Critical Dependency Verification
    print("\n[1/4] Checking environment dependencies...")
    try:
        import cv2
        import torch
        import reportlab
        import pydicom
        print(" -> SUCCESS: Core dependencies (OpenCV, PyTorch, ReportLab, PyDICOM) are loaded correctly.")
    except ImportError as e:
        print(f" -> CRITICAL ERROR: Missing core library dependency: {e}")
        sys.exit(1)

    # 2. Module Validation Framework via PyTest Execution
    print("\n[2/4] Triggering automated unit tests suite via PyTest...")
    try:
        import pytest
        # Automatically detects and executes tests stored within the tests/ directory
        exit_code = pytest.main(["-v", "tests/"])
        if exit_code == 0:
            print(" -> SUCCESS: All standalone unit test cases executed without errors.")
        else:
            print(" -> WARNING: Some unit test cases triggered failures inside the runner framework.")
    except ImportError:
        print(" -> INFO: 'pytest' not installed in runtime framework. Skipping direct suite runner.")

    # 3. Clinical Workflow Simulation Run
    print("\n[3/4] Initializing production workflow example simulation...")
    example_script = os.path.join("examples", "analyze_and_report.py")
    if os.path.exists(example_script):
        # Dispatches the master human-in-the-loop example workflow script
        status = os.system(f"{sys.executable} {example_script}")
        if status == 0:
            print(" -> SUCCESS: End-to-end user and doctor pipeline validation completed safely.")
        else:
            print(" -> ERROR: Clinical workflow sample run generated unexpected execution faults.")
    else:
        print(" -> CRITICAL ERROR: The main file path 'examples/analyze_and_report.py' is missing.")
        sys.exit(1)

    print("\n[4/4] Pipeline verification process finished.")
    print("="*60)

if __name__ == "__main__":
    run_all_tests()
