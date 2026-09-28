import Link from 'next/link'

// Pages link to routes without the base path: Next's Link adds it. Same-page
// anchors and external URLs stay plain anchors.
export function DocLink({
  href,
  title,
  children,
}: {
  href: string
  title?: string
  children: React.ReactNode
}) {
  return href.startsWith('/') && !href.startsWith('//') ? (
    <Link href={href} title={title}>
      {children}
    </Link>
  ) : (
    <a href={href} title={title}>
      {children}
    </a>
  )
}
