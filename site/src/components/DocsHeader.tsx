'use client'

import { usePathname } from 'next/navigation'

import { trailOf } from '@/lib/navigation'

export function DocsHeader({ title }: { title?: string }) {
  let pathname = usePathname()
  let trail = trailOf(pathname)

  if (!title && !trail) {
    return null
  }

  return (
    <header className="mb-9 space-y-1">
      {trail && (
        <p className="font-display text-sm font-medium text-sky-500">
          {trail.family === undefined
            ? trail.group
            : `${trail.family} · ${trail.group}`}
        </p>
      )}
      {title && (
        <h1 className="font-display text-3xl tracking-tight wrap-break-word text-slate-900 dark:text-white">
          {title}
        </h1>
      )}
    </header>
  )
}
