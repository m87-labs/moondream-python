import os
import sys

# In one-file mode, PyInstaller extracts the bundled files to sys._MEIPASS.
# Otherwise, for a non-one-file build, we can fall back to the directory of the executable.
if hasattr(sys, "_MEIPASS"):
    bundle_dir = sys._MEIPASS
else:
    bundle_dir = os.path.dirname(sys.executable)

# The actual name of the bundled libvips file as placed by --add-binary "/usr/lib/x86_64-linux-gnu/libvips.so.42.14.1:."
libvips_actual = os.path.join(bundle_dir, "libvips.so.42.14.1")

# The name pyvips expects to load
libvips_expected = os.path.join(bundle_dir, "libvips.so.42")

# If 'libvips.so.42' doesn't exist but 'libvips.so.42.14.1' does, create a symlink
if not os.path.exists(libvips_expected) and os.path.exists(libvips_actual):
    os.symlink(os.path.abspath(libvips_actual), libvips_expected)

# Ensure our bundle directory is on LD_LIBRARY_PATH
current_ld = os.environ.get("LD_LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = bundle_dir + (":" + current_ld if current_ld else "")

# (Optional) debug print to confirm presence and naming
# print("DEBUG: Contents of bundle_dir:", os.listdir(bundle_dir))
# print("DEBUG: LD_LIBRARY_PATH:", os.environ["LD_LIBRARY_PATH"])