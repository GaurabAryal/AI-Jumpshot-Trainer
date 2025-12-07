"""Main application entry point."""
import sys
import os
import warnings
import logging
from pathlib import Path

# Suppress MediaPipe/TensorFlow warnings BEFORE any imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 3 = suppress all TF logs
os.environ['GLOG_minloglevel'] = '2'  # Suppress glog (used by MediaPipe)
os.environ['MP_VERBOSE'] = '0'  # Suppress MediaPipe verbose output

# Suppress OpenCV warnings
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'

# Suppress Python warnings
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

# Suppress macOS AVCapture warnings by filtering stderr
class FilteredStderr:
    """Filter stderr to suppress known warnings."""
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
        self.filtered_patterns = [
            b'AVCaptureDeviceTypeExternal',
            b'OpenCV:',
            b'WARNING: All log messages before absl::InitializeLog()',
            b'Feedback manager requires',
            b'inference_feedback_manager',
            b'GL version:',
            b'Created TensorFlow Lite',
        ]
    
    def write(self, data):
        if isinstance(data, str):
            data = data.encode('utf-8', errors='ignore')
        # Only write if it doesn't match filtered patterns
        if not any(pattern in data for pattern in self.filtered_patterns):
            self.original_stderr.write(data.decode('utf-8', errors='ignore') if isinstance(data, bytes) else data)
    
    def flush(self):
        self.original_stderr.flush()
    
    def __getattr__(self, name):
        return getattr(self.original_stderr, name)

# Apply stderr filtering
sys.stderr = FilteredStderr(sys.stderr)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from PyQt6.QtWidgets import QApplication
from src.gui.main_window import MainWindow


def main():
    """Run the application."""
    print("[Main] Starting Basketball Trainer application...")
    sys.stdout.flush()
    
    # Check if we're in a headless environment (Linux/Unix)
    if 'DISPLAY' not in os.environ and sys.platform != 'darwin':
        print("[Main] ERROR: No DISPLAY environment variable set. Cannot create GUI.")
        print("[Main] Please run this application in an environment with GUI access.")
        sys.stdout.flush()
        sys.exit(1)
    
    try:
        # On macOS, check if we can access the display
        if sys.platform == 'darwin':
            # Try to detect if we're in a headless environment
            # This is a best-effort check
            pass
        
        app = QApplication(sys.argv)
        
        # Check if QApplication was created successfully
        if app is None:
            print("[Main] ERROR: Failed to create QApplication")
            sys.stdout.flush()
            sys.exit(1)
        
        # Test if we can create a simple widget (this will fail if no display)
        try:
            from PyQt6.QtWidgets import QWidget
            test_widget = QWidget()
            test_widget.close()
            del test_widget
        except Exception as e:
            print(f"[Main] ERROR: Cannot create GUI widgets - no display available")
            print(f"[Main] Error details: {e}")
            print("[Main] Please run this application in Terminal.app or ensure GUI access is available")
            sys.stdout.flush()
            sys.exit(1)
        
        print("[Main] Creating main window...")
        sys.stdout.flush()
        
        try:
            window = MainWindow()
            if window is None:
                print("[Main] ERROR: Failed to create MainWindow")
                sys.stdout.flush()
                sys.exit(1)
            
            window.show()
            print("[Main] Application started successfully")
            sys.stdout.flush()
            
            exit_code = app.exec()
            print(f"[Main] Application exiting with code {exit_code}")
            sys.stdout.flush()
            sys.exit(exit_code)
        except Exception as e:
            print(f"[Main] ERROR creating window: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            sys.exit(1)
    except SystemExit:
        raise
    except Exception as e:
        print(f"[Main] ERROR starting application: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.exit(1)


if __name__ == "__main__":
    main()

