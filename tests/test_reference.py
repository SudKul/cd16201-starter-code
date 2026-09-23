"""Reference infrastructure with arbitrary CSV fixtures, no cleaning implementation."""
import pytest

from components.init_reference import initialize_reference, load_reference
from components.local_pipeline import checksum, create_run, stage


def cleaned_fixture(root, text):
    run = create_run(root, {})
    with stage(root, run, 'basic_cleaning') as record:
        source = run / 'clean.csv'
        source.write_text(text)
        record['outputs']['clean'] = source
    return run, source


def test_reference_is_fixed_across_later_runs_and_rejects_overwrite(tmp_path):
    first, source = cleaned_fixture(tmp_path, 'id,value\n1,example\n')
    metadata = initialize_reference(tmp_path, source, first)
    baseline = load_reference(tmp_path)
    original = baseline.read_bytes()
    second, other = cleaned_fixture(tmp_path, 'id,value\n2,other\n')
    with pytest.raises(FileExistsError):
        initialize_reference(tmp_path, other, second)
    assert baseline.read_bytes() == original
    assert checksum(baseline) == metadata['reference']['sha256']
    assert metadata['source_run_id'] == first.name


def test_missing_reference_and_wrong_source_fail_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match='Initialize it explicitly'):
        load_reference(tmp_path)
    first, source = cleaned_fixture(tmp_path, 'id,value\n1,example\n')
    second, other = cleaned_fixture(tmp_path, 'id,value\n2,other\n')
    with pytest.raises(ValueError, match='completed cleaning output'):
        initialize_reference(tmp_path, source, second)
    initialize_reference(tmp_path, source, first)
    load_reference(tmp_path).write_text('changed')
    with pytest.raises(ValueError, match='checksum mismatch'):
        load_reference(tmp_path)
