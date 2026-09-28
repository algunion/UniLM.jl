import withMarkdoc from '@markdoc/next.js'

import withPreprocessedPages from './src/markdoc/preprocess.mjs'
import withSearch from './src/markdoc/search.mjs'

// Where the static export is served (/UniLM.jl/dev, /UniLM.jl/v0.22.0, …) is a
// build setting: pages hold root-relative routes and Next.js adds the prefix.
const basePath = process.env.DOCS_BASE_PATH ?? ''
const docsVersion = process.env.NEXT_PUBLIC_DOCS_VERSION ?? 'dev'

if (!/^(dev|stable|v\d+\.\d+\.\d+)$/.test(docsVersion)) {
  throw new Error(
    `NEXT_PUBLIC_DOCS_VERSION must be dev, stable or a version like v0.22.0; got ${JSON.stringify(docsVersion)}`,
  )
}

/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  basePath,
  trailingSlash: true,
  images: { unoptimized: true },
  env: {
    NEXT_PUBLIC_BASE_PATH: basePath,
    NEXT_PUBLIC_DOCS_VERSION: docsVersion,
  },
  pageExtensions: ['js', 'jsx', 'md', 'ts', 'tsx'],
}

// @markdoc/next.js compiles a page with Next's default SWC loader, which does
// not treat it as a React Server Component, so Next.js rejects the page's
// `metadata` export. Compile pages in the server-components layer ('rsc'), as
// Next.js compiles its own app code there.
function withServerComponentPages(nextConfig) {
  return Object.assign({}, nextConfig, {
    webpack(config, options) {
      config = nextConfig.webpack(config, options)
      let swc = options.defaultLoaders.babel
      let index = config.module.rules.findIndex(
        (rule) =>
          rule?.use?.[0] === swc &&
          rule.use.some((entry) => entry?.loader?.includes('@markdoc/next.js')),
      )
      if (index === -1) {
        throw new Error('next.config.mjs: the @markdoc/next.js rule is missing')
      }
      let { use, ...rule } = config.module.rules[index]
      let serverSwc = {
        ...swc,
        options: { ...swc.options, bundleLayer: 'rsc', esm: true },
      }
      config.module.rules[index] = {
        ...rule,
        oneOf: [
          { issuerLayer: 'rsc', use: [serverSwc, ...use.slice(1)] },
          { use },
        ],
      }
      return config
    },
  })
}

export default withSearch(
  withPreprocessedPages(
    withServerComponentPages(
      withMarkdoc({
        schemaPath: './src/markdoc',
        // `metadata` turns a page's `nextjs.metadata` frontmatter into its <title>
        nextjsExports: ['metadata', 'revalidate'],
      })(nextConfig),
    ),
  ),
)
