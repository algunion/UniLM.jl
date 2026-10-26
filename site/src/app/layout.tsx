import { type Metadata } from 'next'
import { Inter } from 'next/font/google'
import localFont from 'next/font/local'
import clsx from 'clsx'

import { Providers } from '@/app/theme-providers'
import { Layout } from '@/components/Layout'
import { RenameRetiredAnchor } from '@/components/RenameRetiredAnchor'
import { anchorDestination } from '@/lib/anchors'

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
  description: 'One Julia API for language models and decision models.',
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

// A reader who opens a page with the id of an anchor on another page (an old
// link, a bookmark) moves there before the page hydrates. A retired id whose
// successor is on this page is renamed after hydration instead
// (RenameRetiredAnchor), since Next.js writes back the URL it loaded with as it
// hydrates.
function forwardMovedAnchor(
  base: string,
  destination: typeof anchorDestination,
) {
  addEventListener('DOMContentLoaded', () => {
    destination(base).then((to) => {
      if (to !== undefined && !to.here) {
        location.replace(`${base}${to.route}#${encodeURIComponent(to.id)}`)
      }
    })
  })
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
        <script
          dangerouslySetInnerHTML={{
            __html: `(${forwardMovedAnchor})(${JSON.stringify(basePath)}, ${anchorDestination})`,
          }}
        />
      </head>
      <body className="flex min-h-full bg-white dark:bg-slate-900">
        <Providers>
          <Layout>{children}</Layout>
        </Providers>
        <RenameRetiredAnchor />
      </body>
    </html>
  )
}
