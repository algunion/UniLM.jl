'use client'

import { useEffect } from 'react'

import { type AnchorLookup } from '@/lib/anchors'

// A retired id whose successor is on this page: once the page has hydrated, the
// hash names the successor (Next.js writes back the URL it loaded with as it
// hydrates). The destination is the lookup the page's head started
// (app/layout.tsx). replaceState with a null state is the form Next.js
// documents: its router takes the new URL. It moves nothing, so the successor
// is then scrolled into view, as a jump to its anchor would.
export function RenameRetiredAnchor() {
  useEffect(() => {
    let shared: Window & AnchorLookup = window
    shared.anchorLookup?.then((to) => {
      if (to?.here) {
        history.replaceState(null, '', `#${encodeURIComponent(to.id)}`)
        document.getElementById(to.id)?.scrollIntoView()
      }
    })
  }, [])

  return null
}
