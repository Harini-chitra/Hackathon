from dotenv import load_dotenv
import os
from pathlib import Path

def find_dotenv():
    """Try to find .env file in various locations"""
    # 1. Check current working directory
    cwd_path = Path.cwd() / '.env'
    if cwd_path.exists():
        return cwd_path
        
    # 2. Check script directory (project root)
    script_dir = Path(__file__).parent.parent.parent  # Go up to project root
    script_path = script_dir / '.env'
    if script_path.exists():
        return script_path
        
    # 3. Check home directory
    home_path = Path.home() / '.env'
    if home_path.exists():
        return home_path
        
    # 4. Return the original path as fallback
    return Path(r'C:\Onedrive\Desktop\Project phase1\WEB APPLICATION\qkd-webapp-main\qkd_webapp\.env')

# Find and load .env file
ENV_PATH = find_dotenv()

# For debugging
print("\n=== Environment Configuration ===")
print(f"Using .env file at: {ENV_PATH}")
print(f"File exists: {ENV_PATH.exists()}")

# Try to read the file contents (for debugging)
try:
    if ENV_PATH.exists():
        file_content = ENV_PATH.read_text()
        print(f"File contents start: {file_content[:50]}...")
        print(f"File length: {len(file_content)} characters")
    else:
        print("No .env file found in any standard location")
except Exception as e:
    print(f"Error reading .env file: {e}")

# Load environment variables from .env file if it exists
if ENV_PATH.exists():
    load_dotenv(ENV_PATH, override=True)
    print("Successfully loaded .env file")
else:
    print("WARNING: No .env file found in any standard location")
    print("Searched in:")
    print(f"1. Current directory: {Path.cwd() / '.env'}")
    print(f"2. Project root: {Path(__file__).parent.parent.parent / '.env'}")
    print(f"3. Home directory: {Path.home() / '.env'}")
    print("4. Fallback location:", str(Path('C:\\Onedrive\\Desktop\\Project phase1\\WEB APPLICATION\\qkd-webapp-main\\qkd_webapp\\.env')))
    print("\nPlease create a .env file in one of these locations with your SUPERSTAQ_API_KEY")

# Debug: Show if our key is loaded (showing only first 4 chars for security)
API_KEY = os.getenv('SUPERSTAQ_API_KEY', '')
print(API_KEY)
print(f"SUPERSTAQ_API_KEY loaded: {'Yes' + (' (length: ' + str(len(API_KEY)) + ')' if API_KEY else ' (empty)')}")
if not API_KEY:
    print("ERROR: SUPERSTAQ_API_KEY is required but not found in environment variables")
    print("Please create a .env file in the project root with:")
    print("SUPERSTAQ_API_KEY=your_api_key_here")

class Config:
    # Superstaq Configuration
    SUPERSTAQ_API_KEY = API_KEY
    print(f"SUPERSTAQ_API_KEY: {SUPERSTAQ_API_KEY}")  # Debug line

    @classmethod
    def validate(cls):
        """Validate that all required environment variables are set."""
        required_vars = {
            'SUPERSTAQ_API_KEY': cls.SUPERSTAQ_API_KEY,
        }
        
        missing = [name for name, value in required_vars.items() if not value]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

# Validate configuration on import
Config.validate()
