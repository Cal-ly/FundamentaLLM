#!/usr/bin/env python3
"""
Test script to verify math rendering is working.
This creates a simple test page with various math examples.
"""

from pathlib import Path


def create_test_page():
    """Create a test page with math examples."""
    test_content = """# Math Rendering Test

This page tests various mathematical notation to ensure proper rendering.

## Inline Math

Here's some inline math: $E = mc^2$ and $a^2 + b^2 = c^2$.

The quadratic formula is $x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$.

## Display Math (Block)

### Simple Equation

$$f(x) = x^2 + 2x + 1$$

### Summation

$$\\sum_{i=1}^{n} i = \\frac{n(n+1)}{2}$$

### Fraction and Square Root

$$\\text{Loss} = -\\frac{1}{N} \\sum_{i=1}^{N} \\log P(x_i | x_{<i})$$

### Matrix

$$\\begin{bmatrix}
a & b \\\\
c & d
\\end{bmatrix}$$

### Complex Expression (from autoregressive.md)

$$\\mathcal{L} = -\\frac{1}{N} \\sum_{i=1}^{N} \\log P(x_i | x_{<i})$$

### Attention Formula (from attention.md)

$$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$

### Positional Encoding (from transformers.md)

$$PE(pos, 2i) = \\sin\\left(\\frac{pos}{10000^{2i/d}}\\right)$$

$$PE(pos, 2i+1) = \\cos\\left(\\frac{pos}{10000^{2i/d}}\\right)$$

## Mixed Content

The loss function $$\\mathcal{L}$$ measures how well our model predicts the next token.

For temperature scaling: $$P(x_i) = \\frac{\\exp(\\text{logit}_i / T)}{\\sum_j \\exp(\\text{logit}_j / T)}$$

## Test Results

If you can see properly formatted mathematical notation above (not raw LaTeX code), then the math rendering is working correctly! ✅
"""
    
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    test_file = project_root / 'pages' / 'math-test.md'
    
    test_file.write_text(test_content)
    print(f"✅ Created test page: {test_file}")
    print(f"   View it at: http://localhost:5173/math-test")
    print()
    print("To test:")
    print("1. Make sure the dev server is running (npm run docs:dev)")
    print("2. Open http://localhost:5173/math-test in your browser")
    print("3. Check if the math equations render properly")
    print()
    print("If the math renders correctly, delete this test file.")


if __name__ == '__main__':
    create_test_page()
