import { defineConfig } from 'vitepress'
import mathjax3 from 'markdown-it-mathjax3'

export default defineConfig({
  title: 'FundamentaLLM',
  description: 'Educational Framework for Learning Language Models from First Principles',
  base: '/FundamentaLLM/',
  ignoreDeadLinks: true,
  head: [
    [
      'script',
      {
        type: 'text/javascript',
        id: 'MathJax-script',
        async: true,
        src: 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'
      }
    ]
  ],
  markdown: {
    config: (md) => {
      md.use(mathjax3)
    }
  },
  themeConfig: {
    logo: '/logo.svg',
    siteTitle: 'FundamentaLLM',
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Guide', link: '/guide/introduction' },
      { text: 'Concepts', link: '/concepts/overview' },
      { text: 'Modules', link: '/modules/overview' },
      { text: 'Tutorials', link: '/tutorials/installation' }
    ],
    sidebar: {
      '/guide/': [
        {
          text: 'Getting Started',
          items: [
            { text: 'Introduction', link: '/guide/introduction' },
            { text: 'Installation', link: '/guide/installation' },
            { text: 'Quick Start', link: '/guide/quick-start' },
            { text: 'Tech Stack', link: '/guide/tech-stack' }
          ]
        },
        {
          text: 'Using FundamentaLLM',
          items: [
            { text: 'CLI Overview', link: '/guide/cli-overview' },
            { text: 'Training Models', link: '/guide/training' },
            { text: 'Generating Text', link: '/guide/generation' },
            { text: 'Evaluation', link: '/guide/evaluation' }
          ]
        },
        {
          text: 'Best Practices',
          items: [
            { text: 'Hyperparameter Tuning', link: '/guide/hyperparameters' },
            { text: 'Data Preparation', link: '/guide/data-prep' },
            { text: 'Troubleshooting', link: '/guide/troubleshooting' }
          ]
        }
      ],
      '/concepts/': [
        {
          text: 'Foundations',
          items: [
            { text: 'Overview', link: '/concepts/overview' },
            { text: 'Transformer Architecture', link: '/concepts/transformers' },
            { text: 'Attention Mechanism', link: '/concepts/attention' },
            { text: 'Positional Encoding', link: '/concepts/positional-encoding' }
          ]
        },
        {
          text: 'NLP Fundamentals',
          items: [
            { text: 'Tokenization', link: '/concepts/tokenization' },
            { text: 'Embeddings', link: '/concepts/embeddings' },
            { text: 'Language Modeling', link: '/concepts/language-modeling' },
            { text: 'Autoregressive Generation', link: '/concepts/autoregressive' }
          ]
        },
        {
          text: 'Training & Optimization',
          items: [
            { text: 'Loss Functions', link: '/concepts/losses' },
            { text: 'Optimization Algorithms', link: '/concepts/optimization' },
            { text: 'Learning Rate Scheduling', link: '/concepts/scheduling' },
            { text: 'Gradient Management', link: '/concepts/gradients' }
          ]
        }
      ],
      '/modules/': [
        {
          text: 'Module Guide',
          items: [
            { text: 'Overview', link: '/modules/overview' },
            { text: 'Project Structure', link: '/modules/structure' }
          ]
        },
        {
          text: 'Core Components',
          items: [
            { text: 'Models', link: '/modules/models' },
            { text: 'Tokenizers', link: '/modules/tokenizers' },
            { text: 'Data Pipeline', link: '/modules/data' },
            { text: 'Training Pipeline', link: '/modules/training' }
          ]
        },
        {
          text: 'Advanced Modules',
          items: [
            { text: 'Generation', link: '/modules/generation' },
            { text: 'Evaluation', link: '/modules/evaluation' },
            { text: 'Configuration', link: '/modules/config' },
            { text: 'CLI', link: '/modules/cli' }
          ]
        }
      ],
      '/tutorials/': [
        {
          text: 'Step-by-Step Tutorials',
          items: [
            { text: 'Installation', link: '/tutorials/installation' },
            { text: 'Your First Model', link: '/tutorials/first-model' },
            { text: 'Training Deep Dive', link: '/tutorials/training-deep-dive' },
            { text: 'Advanced Generation', link: '/tutorials/advanced-generation' },
            { text: 'Custom Datasets', link: '/tutorials/custom-datasets' }
          ]
        }
      ]
    },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/your-org/fundamentallm' }
    ],
    footer: {
      message: 'Released under the AGPL-3.0 License',
      copyright: 'Copyright © 2024-2026 FundamentaLLM Contributors'
    }
  }
})
