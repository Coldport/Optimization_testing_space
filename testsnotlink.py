import ctypes
import os

# Add the directory containing the DLL to the search path
os.add_dll_directory(r"C:\Windows\System32")

# Load the DLL
try:
    snopt = ctypes.CDLL(r"C:\Windows\System32\snopt7.dll")
    print("SNOPT loaded successfully!")
except Exception as e:
    print(f"Failed to load SNOPT: {e}")