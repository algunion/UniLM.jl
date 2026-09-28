import { type Metadata } from 'next'
import { Inter } from 'next/font/google'
import localFont from 'next/font/local'
import clsx from 'clsx'

import { Providers } from '@/app/providers'
import { Layout } from '@/components/Layout'

import '@/styles/tailwind.css'

const inter = Inter({
  subsets: ['latin'],
  display: 'swap',
  variable: '--font-inter',
})

// Use local version of Lexend so that we can use OpenType features
const lexend = localFont({
  src: '../fonts/lexend.woff2',
  display: 'swap',
  variable: '--font-lexend',
})

// Manual pages set an absolute title (src/markdoc/preprocess.mjs).
export const metadata: Metadata = {
  title: {
    template: '%s - UniLM.jl',
    default: 'UniLM.jl',
  },
  description: 'A unified Julia interface for large language models.',
}

const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? ''

// A version's export is also served under Documenter's aliases for it
// (`stable`, `v0.22` and `v0` link to `v0.22.0`), but Next.js reads the route
// after its base path and fails to hydrate a page opened elsewhere. Before
// hydration, such a page moves to its route under the base path: the path's
// first segments, as many as the base path has, become the base path.
function moveUnderBasePath(base: string) {
  let path = location.pathname
  if (path === base || path.startsWith(`${base}/`)) return
  let rest = path.split('/').slice(base.split('/').length)
  location.replace(
    `${base}/${rest.join('/')}${location.search}${location.hash}`,
  )
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html
      lang="en"
      className={clsx('h-full antialiased', inter.variable, lexend.variable)}
      suppressHydrationWarning
    >
      <head>
        {basePath !== '' && (
          <script
            dangerouslySetInnerHTML={{
              __html: `(${moveUnderBasePath})(${JSON.stringify(basePath)})`,
            }}
          />
        )}
      </head>
      <body className="flex min-h-full bg-white dark:bg-slate-900">
        <Providers>
          <Layout>{children}</Layout>
        </Providers>
      </body>
    </html>
  )
}
