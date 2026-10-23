'use client'

import { useEffect } from 'react'

import { anchorDestination } from '@/lib/anchors'

// A retired id whose successor is on this page: once the page has hydrated, the
// hash names the successor. Earlier, Next.js would write back the URL it loaded
// with as it hydrates. replaceState keeps Next.js's history state and moves
// nothing, so the successor is then scrolled into view, as a jump to its anchor
// would.
export function RenameRetiredAnchor() {
  useEffect(() => {
    anchorDestination(process.env.NEXT_PUBLIC_BASE_PATH ?? '').then((to) => {
      if (to?.here) {
        history.replaceState(history.state, '', `#${encodeURIComponent(to.id)}`)
        document.getElementById(to.id)?.scrollIntoView()
      }
    })
  }, [])

  return null
}
