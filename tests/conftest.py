import pytest

from checkpoints import write_kolibri1_checkpoint


def pytest_addoption(parser):
    parser.addoption(
        "--tokenizer",
        default="gpt2",
        help="Tokenizer for GPU model loading tests. Any tokenizer with a "
        "vocab smaller than the test configs' vocab_size works.",
    )


@pytest.fixture(scope="session")
def tokenizer(request):
    return request.config.getoption("--tokenizer")


@pytest.fixture(scope="session")
def register_plugin():
    from aleph_alpha_inference import register

    register()
    # Plugins are loaded once per process, but every vLLM process loads
    # them, so registering again must be a no-op.
    register()


@pytest.fixture(scope="session")
def kolibri1_model_dir(tmp_path_factory) -> str:
    return write_kolibri1_checkpoint(str(tmp_path_factory.mktemp("kolibri1")))
