import pytest
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from nlacp.extraction.resource_extractor import extract_resource, extract_resources

def test_extract_resource_single():
    # Câu có 1 object, kiểm tra hàm extract_resource cũ vẫn hoạt động
    sentence = "A student can view their own grades."
    resource = extract_resource(sentence)
    
    assert resource is not None
    assert resource.label == "grade"
    assert resource.qualifier in ("own", "the")

def test_extract_resources_multiple():
    # Câu có nhiều objects nối bằng "and", kiểm tra hàm extract_resources mới
    sentence = "Students can view their own grades and transcripts."
    resources = extract_resources(sentence)
    
    assert resources is not None
    assert len(resources) == 2
    
    labels = {res.label for res in resources}
    assert "grade" in labels
    assert "transcript" in labels
    
    # Check that qualifier is correctly extracted for both if they share context
    for res in resources:
        assert res.qualifier in ("own", "the")
