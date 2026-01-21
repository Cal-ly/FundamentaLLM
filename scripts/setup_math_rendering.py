#!/usr/bin/env python3
"""
Configure VitePress to support mathematical notation rendering using markdown-it-mathjax3.
This script will:
1. Install the necessary npm package
2. Update the VitePress config to enable math rendering
"""

import subprocess
import json
from pathlib import Path
import sys


def run_command(cmd, cwd=None):
    """Run a shell command and return the result."""
    result = subprocess.run(
        cmd, 
        shell=True, 
        cwd=cwd, 
        capture_output=True, 
        text=True
    )
    return result


def main():
    """Main function to configure math rendering in VitePress."""
    # Get the project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    pages_dir = project_root / 'pages'
    
    if not pages_dir.exists():
        print(f"Error: pages directory not found at {pages_dir}")
        sys.exit(1)
    
    print("=" * 60)
    print("Configuring VitePress for Mathematical Notation")
    print("=" * 60)
    print()
    
    # Step 1: Install markdown-it-mathjax3
    print("Step 1: Installing markdown-it-mathjax3...")
    result = run_command("npm install -D markdown-it-mathjax3", cwd=pages_dir)
    
    if result.returncode == 0:
        print("✅ Successfully installed markdown-it-mathjax3")
    else:
        print("❌ Failed to install markdown-it-mathjax3")
        print(result.stderr)
        sys.exit(1)
    
    print()
    
    # Step 2: Update VitePress config
    print("Step 2: Updating VitePress configuration...")
    config_file = pages_dir / '.vitepress' / 'config.js'
    
    if not config_file.exists():
        print(f"❌ Config file not found at {config_file}")
        sys.exit(1)
    
    # Read current config
    config_content = config_file.read_text()
    
    # Check if markdown-it-mathjax3 is already configured
    if 'markdown-it-mathjax3' in config_content:
        print("⚠️  markdown-it-mathjax3 is already configured in config.js")
        print("   Please check if it's correctly set up.")
    else:
        # Add the import at the top
        if "import mathjax3 from 'markdown-it-mathjax3'" not in config_content:
            # Add import after the first import line
            lines = config_content.split('\n')
            for i, line in enumerate(lines):
                if line.startswith('import '):
                    lines.insert(i + 1, "import mathjax3 from 'markdown-it-mathjax3'")
                    break
            config_content = '\n'.join(lines)
        
        # Add markdown config to the defineConfig object
        # Find where to insert the markdown config
        if 'markdown:' not in config_content:
            # Insert before themeConfig
            config_content = config_content.replace(
                '  themeConfig: {',
                '''  markdown: {
    config: (md) => {
      md.use(mathjax3)
    }
  },
  themeConfig: {'''
            )
        
        # Write updated config
        config_file.write_text(config_content)
        print("✅ Updated VitePress configuration")
    
    print()
    
    # Step 3: Verify the configuration
    print("Step 3: Verifying configuration...")
    config_content = config_file.read_text()
    
    has_import = "markdown-it-mathjax3" in config_content
    has_markdown_config = "markdown:" in config_content and "md.use(mathjax3)" in config_content
    
    if has_import and has_markdown_config:
        print("✅ Configuration verified successfully!")
    else:
        print("⚠️  Configuration may be incomplete:")
        if not has_import:
            print("   - Missing import statement")
        if not has_markdown_config:
            print("   - Missing markdown configuration")
    
    print()
    print("=" * 60)
    print("Math Rendering Setup Complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Test the math rendering with: cd pages && npm run docs:dev")
    print("2. Visit a page with math notation (e.g., concepts/autoregressive)")
    print("3. Math should render with proper formatting")
    print()
    print("Note: If math doesn't render, you may need to:")
    print("- Add MathJax CSS to the head")
    print("- Or use an alternative plugin like markdown-it-katex")
    print()


if __name__ == '__main__':
    main()
