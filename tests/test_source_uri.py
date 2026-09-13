from recall.core.source_uri import display_basename, is_junk_source_uri


def test_is_junk_source_uri_detects_tempfile_names():
    assert is_junk_source_uri("tmp1nk3baf4.pdf")
    assert is_junk_source_uri("/private/var/folders/xx/T/tmp1nk3baf4.pdf")
    assert is_junk_source_uri("/tmp/upload.pdf")


def test_is_junk_source_uri_keeps_real_documents():
    assert not is_junk_source_uri("employee_handbook.docx")
    assert not is_junk_source_uri("/Users/me/docs/Ai Assurance.pdf")
    assert not is_junk_source_uri("k8s-guide.md")


def test_display_basename():
    assert display_basename("/path/to/policy.md") == "policy.md"
    assert display_basename("policy.md") == "policy.md"
