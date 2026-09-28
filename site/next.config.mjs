import withMarkdoc from '@markdoc/next.js'

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

export default withSearch(
  withMarkdoc({ schemaPath: './src/markdoc', nextjsExports: ['revalidate'] })(
    nextConfig,
  ),
)
