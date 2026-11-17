"""Expose les tests existants (fonctions main()) à pytest."""
from tests import test_pipeline, test_translate


def test_pipeline_end_to_end():
    assert test_pipeline.main() == 0


def test_translate_without_key():
    assert test_translate.main() == 0
