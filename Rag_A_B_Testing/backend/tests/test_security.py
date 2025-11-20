from app import clean_text_artifacts
import sys
import os
import pytest

# Add parent directory to path so we can import app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_clean_text_artifacts():
    """Test that trailing numbers are removed from names."""
    raw_input = "Does Donetta1 Bradtke547 have allergies?"
    expected = "Does Donetta Bradtke have allergies?"
    assert clean_text_artifacts(raw_input) == expected


def test_clean_text_artifacts_no_change():
    """Test that normal text is left alone."""
    raw_input = "Patient has hypertension."
    assert clean_text_artifacts(raw_input) == raw_input

# Note: We skip testing 'redact_pii' in CI because downloading
# the large spacy model in GitHub Actions takes too long for a quick demo.
