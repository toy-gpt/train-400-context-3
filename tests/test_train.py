"""
tests/test_imports.py

REQ: Verify that the package can be imported.
WHY: Minimal correctness requirement.

REQ: Verify that demo modules can execute without raising exceptions.
WHY: Each stage should be able to be run independently on any platform.
"""

import importlib

PACKAGE_NAME = "toy_gpt_train"

DEMO_MODULES = [
    "a_tokenizer",
    "b_vocab",
    "c_model",
    "d_train",
    "e_infer",
]


def test_package_imports() -> None:
    """Test that the package itself can be imported."""
    module = importlib.import_module(PACKAGE_NAME)
    assert module is not None


def test_demo_modules_run() -> None:
    """
    Test that each demo module can be imported and run.

    OBS: This test does not validate outputs.
         It only checks that demo entry points execute without error.
    """
    for module_name in DEMO_MODULES:
        module_path = f"{PACKAGE_NAME}.{module_name}"
        module = importlib.import_module(module_path)

        # If a module defines main(), it should be runnable
        if hasattr(module, "main"):
            module.main()
