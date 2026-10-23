import { type Metadata } from 'next'
import { Inter } from 'next/font/google'
import localFont from 'next/font/local'
import clsx from 'clsx'

import { Providers } from '@/app/theme-providers'
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

// <base>/anchor-aliases.json and <base>/anchors.json
type Aliases = Record<string, string>
type Anchors = Record<string, Array<string>>

// A reader who opens a page with the id of an anchor that is on another page
// (an old link, a bookmark) is moved there, when the id names nothing on the
// page. anchors.json maps each anchor id to the routes that define it, and
// anchor-aliases.json a retired id to its successor (no aliases when it cannot
// be fetched). The reader stays when this route defines the id, the hash then
// naming the successor, and when several routes or none do; otherwise they move
// to the one route that defines it, so a move never repeats.
function forwardMovedAnchor(base: string) {
  addEventListener('DOMContentLoaded', () => {
    let id: string
    try {
      id = decodeURIComponent(location.hash.slice(1))
    } catch {
      return // a hash that does not decode names no anchor
    }
    if (id === '' || document.getElementById(id) !== null) return
    let read = (name: string) =>
      fetch(`${base}/${name}`).then((response) => {
        if (!response.ok) throw new Error(`${name}: HTTP ${response.status}`)
        return response.json()
      })
    Promise.all([
      read('anchor-aliases.json').catch(() => ({})),
      read('anchors.json'),
    ]).then(([aliases, anchors]: [Aliases, Anchors]) => {
      let to = Object.hasOwn(aliases, id) ? aliases[id] : id
      let routes = Object.hasOwn(anchors, to) ? anchors[to] : []
      let path = location.pathname.slice(base.length)
      if (routes.includes(path.endsWith('/') ? path : `${path}/`)) {
        if (to !== id) location.replace(`#${encodeURIComponent(to)}`)
      } else if (routes.length === 1) {
        location.replace(`${base}${routes[0]}#${encodeURIComponent(to)}`)
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
            __html: `(${forwardMovedAnchor})(${JSON.stringify(basePath)})`,
          }}
        />
      </head>
      <body className="flex min-h-full bg-white dark:bg-slate-900">
        <Providers>
          <Layout>{children}</Layout>
        </Providers>
      </body>
    </html>
  )
}
