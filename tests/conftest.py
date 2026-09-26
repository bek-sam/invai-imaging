import os
import tempfile

# Tests run against the local-directory storage backend; set before app.config is imported.
os.environ["STORAGE"] = "local"
os.environ["LOCAL_STORAGE_DIR"] = tempfile.mkdtemp(prefix="invai-imaging-test-")
# The API tests use /sample-art to make fixtures; the dev-only gating is tested on its own.
os.environ["IMAGING_DEV_ENDPOINTS"] = "true"
