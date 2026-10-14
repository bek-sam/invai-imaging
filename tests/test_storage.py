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
        assert head["ContentDisposition"] == (
            'attachment; filename="SH-1042_DB-1001; evil.png"; '
            "filename*=UTF-8''SH-1042_DB-1001%3B%20evil.png"
        )

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


def test_content_disposition_drops_lone_surrogates_keeps_whole_characters():
    from app.storage import content_disposition_for

    out = content_disposition_for("SH-1\ud800_\udfff-é-ü-\U0001f600-ñ.png")
    assert out == (
        'attachment; filename="SH-1_-_-_-_-_.png"; '
        "filename*=UTF-8''SH-1_-%C3%A9-%C3%BC-%F0%9F%98%80-%C3%B1.png"
    )
    assert out.isascii()
    # a valid surrogate pair is one emoji (one "_" in the fallback), not two lone halves
    assert "%F0%9F%98%80" in content_disposition_for("a\U0001f600b")
    assert 'filename="a_b"' in content_disposition_for("a\U0001f600b")


def test_content_disposition_is_pure_ascii_and_exact():
    from app.storage import content_disposition_for

    h = content_disposition_for("SH-9 é.png")
    assert h.isascii()
    assert h == "attachment; filename=\"SH-9 _.png\"; filename*=UTF-8''SH-9%20%C3%A9.png"


def test_content_disposition_strips_every_c0_control_and_del_and_c1():
    from app.storage import content_disposition_for

    for cp in [*range(0x20), 0x7F, *range(0x80, 0xA0)]:
        h = content_disposition_for(f"a{chr(cp)}b.png")
        assert h == ("attachment; filename=\"ab.png\"; filename*=UTF-8''ab.png"), hex(cp)
        assert h.isascii()


def test_content_disposition_strips_quote_and_backslash_including_trailing():
    from app.storage import content_disposition_for

    assert content_disposition_for('a"b\\c.png') == (
        "attachment; filename=\"abc.png\"; filename*=UTF-8''abc.png"
    )
    # a trailing backslash would escape the closing quote of the quoted-string
    h = content_disposition_for("sheet\\")
    assert h == "attachment; filename=\"sheet\"; filename*=UTF-8''sheet"
    assert '\\"' not in h
    assert "\\" not in content_disposition_for("\\\\\\")


def test_s3_upload_passes_the_exact_header_to_the_boto_client(tmp_path):
    from unittest.mock import MagicMock

    src = tmp_path / "a.png"
    src.write_bytes(b"x")
    s = S3Storage.__new__(S3Storage)
    s.client = MagicMock()
    s.bucket = "b"
    s.upload(src, "a/b.png", download_filename="SH-1 é\\")
    (_, bucket, key), kw = s.client.upload_file.call_args
    assert (bucket, key) == ("b", "a/b.png")
    cd = kw["ExtraArgs"]["ContentDisposition"]
    assert cd == "attachment; filename=\"SH-1 _\"; filename*=UTF-8''SH-1%20%C3%A9"
    assert cd.isascii()


def test_s3_upload_with_lone_surrogate_name_puts_cleanly_against_stub(tmp_path):
    # S-86: botocore raised on a lone surrogate in the header; the put must now succeed.
    from tests.s3_stub import StubS3

    src = tmp_path / "a.png"
    src.write_bytes(b"png-bytes")
    with StubS3(tmp_path) as stub:
        cfg = Settings(
            storage="s3",
            s3_endpoint=stub.endpoint,
            s3_bucket="stub-bucket",
            s3_access_key_id="x",
            s3_secret_access_key="y",
        )
        S3Storage(cfg).upload(src, "a/b.png", download_filename="SH-1\ud83d_é.png")
        puts = stub.puts()
    assert len(puts) == 1
    cd = puts[0]["headers"]["content-disposition"]
    assert cd == "attachment; filename=\"SH-1__.png\"; filename*=UTF-8''SH-1_%C3%A9.png"
    assert cd.isascii()
    assert not any(0xD800 <= ord(c) <= 0xDFFF for c in cd)
