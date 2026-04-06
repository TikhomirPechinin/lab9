import pytest
from ai_utils.text_preprocessing import clean_text, remove_stopwords, stem_text, tokenize_text

def test_clean_text():
    assert clean_text("Hello, World!") == "hello world"
    assert clean_text("  TEST  ") == "test"
    assert clean_text("Привет, Мир!") == "привет мир"

def test_clean_text_empty():
    assert clean_text("") == ""

def test_clean_text_not_string():
    with pytest.raises(TypeError):
        clean_text(123)

def test_remove_stopwords():
    result = remove_stopwords("this is a test sentence")
    assert "this" not in result
    assert "is" not in result
    assert "a" not in result

def test_remove_stopwords_empty():
    assert remove_stopwords("") == ""

def test_stem_text():
    assert stem_text("running quickly") == "run quickli"
    assert stem_text("studies studied") == "studi studi"

def test_tokenize_text():
    assert tokenize_text("hello world") == ["hello", "world"]