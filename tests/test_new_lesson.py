"""Safety checks for the lesson scaffolder."""

from __future__ import annotations

import pytest

from tools.new_lesson import validate_slug


@pytest.mark.parametrize(
    "slug",
    [
        "04_transformer/02_attention_flow",
        "05_gpt/01-kv-cache",
        "04_transformer/02_attention_flow.ipynb",
    ],
)
def test_lesson_slugs_are_normalised(slug):
    assert validate_slug(slug).startswith(("04_transformer/", "05_gpt/"))
    assert not validate_slug(slug).endswith(".ipynb")


@pytest.mark.parametrize("slug", ["../outside", "/tmp/lesson", "Title With Spaces", "", "x//y"])
def test_lesson_slugs_cannot_escape_or_create_ambiguous_paths(slug):
    with pytest.raises(ValueError, match="invalid lesson slug"):
        validate_slug(slug)


@pytest.mark.parametrize("slug", ["attention", "kv-cache", "rope-2"])
def test_page_slugs_are_single_safe_directory_names(slug):
    assert validate_slug(slug, page=True) == slug


@pytest.mark.parametrize("slug", ["../attention", "nested/page", "Bad_Page"])
def test_page_slugs_reject_traversal_and_nested_paths(slug):
    with pytest.raises(ValueError, match="invalid page slug"):
        validate_slug(slug, page=True)
