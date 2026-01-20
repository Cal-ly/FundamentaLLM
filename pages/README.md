# FundamentaLLM Documentation

This folder contains the VitePress-based documentation website for FundamentaLLM.

## Overview

The documentation is built with [VitePress](https://vitepress.dev/), a static site generator built on Vite. It's focused on being educational, explaining both the "how" and "why" of language models and the FundamentaLLM framework.

## Structure

```
pages/
├── .vitepress/
│   ├── config.js          ← VitePress configuration
│   └── dist/              ← Built static site (generated)
├── index.md               ← Home page
├── guide/                 ← Practical how-to documentation
│   ├── introduction.md
│   ├── tech-stack.md
│   ├── installation.md
│   ├── quick-start.md
│   ├── cli-overview.md
│   └── ...
├── concepts/              ← Theoretical explanations
│   ├── overview.md
│   ├── transformers.md
│   └── ...
├── modules/               ← Implementation deep dives
│   ├── overview.md
│   └── ...
├── tutorials/             ← Step-by-step walkthroughs
│   ├── installation.md
│   └── ...
└── package.json          ← Node.js dependencies
```

## Development

### Install Dependencies

```bash
cd pages
npm install
```

### Local Development Server

```bash
npm run docs:dev
```

Visit `http://localhost:5173` to see the site. Changes to markdown files are reflected instantly (hot reload).

### Build for Production

```bash
npm run docs:build
```

This creates a static site in `.vitepress/dist/` that can be deployed anywhere.

### Preview Production Build

```bash
npm run docs:preview
```

## Deployment

The documentation automatically deploys to GitHub Pages when you push to `main` or `develop` branches (see `.github/workflows/docs.yml`).

### Manual Deployment

If you need to deploy manually:

```bash
# Build the site
npm run docs:build

# The built site is in .vitepress/dist/
# Deploy this folder to your hosting provider
```

### GitHub Pages Setup

The repository is configured to deploy from GitHub Actions:

1. ✅ Automatic builds and deploys on push to `main`/`develop`
2. ✅ Triggers on changes to `pages/` folder
3. ✅ Deployed to `https://github.com/your-org/fundamentallm/pages`

To view the live site after deployment, check your repository's Settings → Pages.

## Writing Documentation

### Adding New Pages

1. Create a markdown file in the appropriate folder:
   - `guide/` - Practical how-to content
   - `concepts/` - Theoretical explanations
   - `modules/` - Code/implementation details
   - `tutorials/` - Step-by-step walkthroughs

2. Update `.vitepress/config.js` sidebar to link to your new page

3. Use standard markdown with code blocks:

```markdown
# Page Title

## Section

Some content with **bold** and `code`.

### Code Block

\`\`\`python
print("Hello, FundamentaLLM!")
\`\`\`

### Links

[Another page](./other-page.md)
[External link](https://example.com)
```

### Markdown Features

VitePress supports standard GitHub Flavored Markdown plus:

**Code highlighting with line numbers:**
```markdown
\`\`\`python {2,5}
def hello():
    print("highlighted line")
    print("normal line")
    print("normal line")
    print("highlighted line")
\`\`\`
```

**Callout blocks:**
```markdown
::: info
This is an info box
:::

::: warning
This is a warning
:::

::: danger
This is a danger/error box
:::

::: details
Click to expand hidden content
:::
```

**Math equations (KaTeX):**
```markdown
Inline: $a^2 + b^2 = c^2$

Block:
$$
\frac{1}{1+e^{-x}}
$$
```

### Best Practices

1. **Educational focus** - Explain concepts thoroughly, not just "what works"
2. **Include examples** - Code examples for every feature
3. **Cross-reference** - Link to related concepts and modules
4. **Use visuals** - ASCII diagrams, tables, code blocks
5. **Consistent formatting** - Follow existing style (see other pages)

## Configuration

### Customizing the Theme

Edit `.vitepress/config.js`:

```javascript
// Change title
title: 'FundamentaLLM',

// Change navigation
nav: [
  { text: 'Home', link: '/' },
  { text: 'Guide', link: '/guide/introduction' },
]

// Change sidebar
sidebar: {
  '/guide/': [ ... ]
}

// Social links
socialLinks: [
  { icon: 'github', link: 'https://github.com/...' }
]
```

### Theming

VitePress uses a clean default theme. Customize colors in `.vitepress/config.js` under `themeConfig`.

See [VitePress Theme Config](https://vitepress.dev/reference/site-config#theme-config) for options.

## Troubleshooting

### Port already in use

If `npm run docs:dev` says port 5173 is in use:

```bash
# Specify a different port
npx vitepress dev --port 3000
```

### Node modules not found

```bash
# Reinstall dependencies
rm -rf node_modules package-lock.json
npm install
```

### Build fails

Check for:
- Dead links (fixed with `ignoreDeadLinks: true`)
- Invalid markdown syntax
- Missing code fence endings

Run `npm run docs:build` to see detailed errors.

## CI/CD Integration

### GitHub Actions Workflow

The `.github/workflows/docs.yml` workflow:

1. Triggers on push to `main`/`develop` (only if `pages/` changed)
2. Installs Node.js dependencies
3. Builds the site with `npm run docs:build`
4. Deploys to GitHub Pages

To modify:
- Edit `.github/workflows/docs.yml`
- Change branches, paths, or build commands as needed

## Resources

- [VitePress Documentation](https://vitepress.dev/)
- [Markdown Guide](https://www.markdownguide.org/)
- [GitHub Pages](https://pages.github.com/)

## Contributing

When adding new documentation:

1. Follow the structure (guide/concepts/modules/tutorials)
2. Use educational tone - explain "why" not just "what"
3. Include examples and code blocks
4. Cross-link related topics
5. Test locally with `npm run docs:dev`
6. Push to trigger automatic deployment

## License

Documentation is part of FundamentaLLM and covered by the same AGPL-3.0 license.
