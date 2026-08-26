"""Use the bundled modern SQLite build required by Chroma."""

import sys

# Preserve Ubuntu's default crash-hook initialization when available.
try:
    import apport_python_hook
except ImportError:
    pass
else:
    apport_python_hook.install()

try:
    import pysqlite3
except ImportError:
    pass
else:
    sys.modules["sqlite3"] = pysqlite3
