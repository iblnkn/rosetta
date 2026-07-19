# Sphinx configuration for the Rosetta documentation site.
#
# Build locally:
#   pip install -r doc/requirements.txt
#   sphinx-build -b html doc doc/_build/html
#
# The site follows the ros2_control / Nav2 / MoveIt pattern: Sphinx + MyST
# (content stays markdown), sphinx_rtd_theme, GitHub Pages. Versioned
# builds (/rolling/-style URL prefixes) come later via per-branch CI builds.

project = "Rosetta"
author = "Isaac Blankenau"
copyright = "2026, Isaac Blankenau"

extensions = [
    "myst_parser",
    "sphinx_copybutton",
]

# Strip shell prompts when copying code blocks.
copybutton_prompt_text = r"\$ |>>> "
copybutton_prompt_is_regexp = True

# Gated Hugging Face repos answer 401 to anonymous linkcheck requests.
linkcheck_ignore = [
    r"https://huggingface\.co/datasets/lerobot/droid",
]

myst_enable_extensions = [
    "colon_fence",
]
myst_heading_anchors = 4

source_suffix = {
    ".md": "markdown",
}

exclude_patterns = [
    "_build",
    "ONBOARDING.md",
]

html_theme = "sphinx_rtd_theme"
html_title = "Rosetta"
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 3,
}
