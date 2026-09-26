import boto3
from moto import mock_aws

from app.config import Settings
from app.storage import LocalStorage, NotFoundError, S3Storage


def test_s3_storage_roundtrip(tmp_path):
    with mock_aws():
        cfg = Settings(
            storage="s3",
            s3_endpoint=None,
            s3_bucket="test-bucket",
            s3_access_key_id="x",
            s3_secret_access_key="y",
        )
        s = S3Storage(cfg)
        s.ensure_bucket()
        s.ensure_bucket()  # idempotent
        assert "test-bucket" in [
            b["Name"] for b in boto3.client("s3", region_name="us-east-1").list_buckets()["Buckets"]
        ]
        src = tmp_path / "a.png"
        src.write_bytes(b"png-bytes")
        s.upload(src, "a/b.png")
        s.put_bytes(b"pdf", "a/c.pdf")
        assert s.exists("a/b.png") and not s.exists("a/zzz.png")
        head = s.client.head_object(Bucket="test-bucket", Key="a/c.pdf")
        assert head["ContentType"] == "application/pdf"
        s.download("a/b.png", tmp_path / "out")
        assert (tmp_path / "out").read_bytes() == b"png-bytes"
        try:
            s.download("missing", tmp_path / "m")
            raise AssertionError("expected NotFoundError")
        except NotFoundError:
            pass


def test_s3_storage_sets_content_disposition_from_filename_hint(tmp_path):
    # AC2: the download filename (sheet id + order numbers) travels as ContentDisposition, never
    # the S3 key itself (s3.ts keeps keys opaque/random -- see wave.md's AC2 scope note).
    with mock_aws():
        cfg = Settings(
            storage="s3",
            s3_endpoint=None,
            s3_bucket="test-bucket",
            s3_access_key_id="x",
            s3_secret_access_key="y",
        )
        s = S3Storage(cfg)
        s.ensure_bucket()
        src = tmp_path / "a.png"
        src.write_bytes(b"png-bytes")
        s.upload(src, "a/b.png", download_filename='SH-1042_DB-1001"; evil\r\n.png')
        head = s.client.head_object(Bucket="test-bucket", Key="a/b.png")
        assert head["ContentDisposition"] == 'attachment; filename="SH-1042_DB-1001; evil.png"'

        # no hint -> no ContentDisposition header at all (existing callers are unaffected)
        s.upload(src, "a/c.png")
        head2 = s.client.head_object(Bucket="test-bucket", Key="a/c.png")
        assert "ContentDisposition" not in head2


def test_local_storage_rejects_escaping_keys(tmp_path):
    s = LocalStorage(tmp_path)
    try:
        s.put_bytes(b"x", "../../etc/evil")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
