"""
Script to scrub all markdown cells in the generated notebooks
so that NO en dash, em dash, equals sign, or hyphen appears in any markdown text.
"""

import json
import re

notebook_files = [
    "24i-2013_AbdulAzeem_BAI-5A.ipynb",
    "24i-2013_Abdul_BAI-5A.ipynb"
]

def clean_markdown_text(text):
    # Replacements for specific known terms
    replacements = [
        ("24i-2013", "24i 2013"),
        ("BAI-5A", "BAI 5A"),
        ("ROC-AUC", "ROC AUC"),
        ("K-Means", "K Means"),
        ("k-means", "k means"),
        ("non-linear", "non linear"),
        ("non-convex", "non convex"),
        ("sub-gradient", "sub gradient"),
        ("cost-sensitive", "cost sensitive"),
        ("well-separated", "well separated"),
        ("log-log", "log log"),
        ("digits-1500", "digits 1500"),
        ("one-vs-one", "one vs one"),
        ("one-vs-rest", "one vs rest"),
        ("One-vs-One", "One vs One"),
        ("One-vs-Rest", "One vs Rest"),
        ("+-1000", "plus or minus 1000"),
        ("+1000", "positive 1000"),
        ("-1000", "negative 1000"),
        ("f - f*", "f minus f*"),
        ("f(theta) - f*", "f(theta) minus f*"),
        ("1 - y", "1 minus y"),
        ("1 - p", "1 minus p"),
        ("1 - Specificity", "1 minus Specificity"),
        ("w - u", "w minus u"),
        ("w - (1 / c)", "w minus (1 / c)"),
        ("c * w - u", "c * w minus u"),
        ("f(x) - y", "f(x) minus y"),
        ("p - y", "p minus y"),
        ("||w||2", "squared norm of w"),
        ("||w||^2", "squared norm of w"),
        ("+1", "positive 1"),
        ("-1", "negative 1"),
        ("p = ", "p is "),
        ("f(x) = ", "f(x) evaluates to "),
        ("tau = ", "tau is "),
        ("r = ", "r is "),
        ("C = ", "C is "),
        ("gamma = ", "gamma is "),
        ("K = ", "K is "),
        ("k = ", "k is "),
        ("w.x + b = 0", "w.x + b is 0"),
        ("w . x + b = 0", "w . x + b is 0"),
        (" = ", " is "),
        ("==", "equals"),
        ("<=", "less than or equal to"),
        (">=", "greater than or equal to"),
        ("!=", "not equal to"),
        ("–", " "),
        ("—", " "),
    ]

    for old, new in replacements:
        text = text.replace(old, new)

    # General replacements for any remaining hyphens or equals
    # If there is a hyphen between words, replace with space
    text = re.sub(r'(\w)-(\w)', r'\1 \2', text)
    # If there is an equals sign, replace with is
    text = text.replace('=', ' is ')
    # If there is an en dash or em dash
    text = text.replace('\u2013', ' ').replace('\u2014', ' ')
    # If there are any stray hyphens e.g. at start of lines or lists
    text = re.sub(r'(^|\n)\s*-\s+', r'\1* ', text)
    # Any remaining hyphens
    text = text.replace('-', ' ')

    # Clean up double spaces
    text = re.sub(r' +', ' ', text)
    return text

for nb_file in notebook_files:
    with open(nb_file, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb['cells']:
        if cell['cell_type'] == 'markdown':
            cleaned_source = []
            for line in cell['source']:
                cleaned_line = clean_markdown_text(line)
                cleaned_source.append(cleaned_line)
            cell['source'] = cleaned_source

    with open(nb_file, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

print("Scrubbing complete!")
