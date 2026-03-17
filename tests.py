import unittest
import os
import sys

def run_tests():
    # Ensure the current directory is in python path to find vvtk_dataset
    sys.path.append(os.getcwd())
    
    loader = unittest.TestLoader()
    start_dir = 'tests'
    
    if not os.path.exists(start_dir):
        print(f"Error: Directory '{start_dir}' not found.")
        sys.exit(1)
        
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if not result.wasSuccessful():
        sys.exit(1)

if __name__ == '__main__':
    run_tests()

