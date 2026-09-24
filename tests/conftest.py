import os
import tempfile

# Tests run against the local-directory storage backend; set before app.config is imported.
os.environ["STORAGE"] = "local"
os.environ["LOCAL_STORAGE_DIR"] = tempfile.mkdtemp(prefix="invai-imaging-test-")
