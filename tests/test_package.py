from importlib.metadata import version


def test_version_matches_metadata():
    import aleph_alpha_inference

    assert aleph_alpha_inference.__version__ == version("aleph-alpha-inference")
