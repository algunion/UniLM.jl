import Link from 'next/link'

// Pages link to routes without the base path: Next's Link adds it. A file of
// the manual's assets (/assets/…) is no route, so the router must neither
// prefetch it nor navigate to it: a plain anchor, with the base path. Same-page
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
  if (href.startsWith('/assets/')) {
    return (
      <a
        href={`${process.env.NEXT_PUBLIC_BASE_PATH ?? ''}${href}`}
        title={title}
      >
        {children}
      </a>
    )
  }
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
