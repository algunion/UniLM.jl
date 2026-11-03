// <base>/anchors.json (the docs build: each anchor id a page declares, mapped to
// the routes of the pages that define it) and <base>/anchor-aliases.json
// (docs/redirects.jl: each retired id, mapped to its successor)
export type AnchorDestination = { route: string; id: string; here: boolean }

// The lookup the page's head starts (app/layout.tsx), kept on `window` for
// RenameRetiredAnchor.
export type AnchorLookup = {
  anchorLookup?: Promise<AnchorDestination | undefined>
}

// Where a reader who opened the page at `path` with `hash` belongs when the
// page does not define the hash's id. A retired id becomes its successor, then
// the routes that define the id decide: this page when it is one of them
// (`here`, the hash then naming the successor), else the one route that
// defines it. Resolves to undefined when the reader stays: the hash is empty,
// does not decode or names an element here, the path is not under `base`,
// several routes or none define the id, or a map cannot be read or is not of
// the shape above (logged). No aliases file (a 404, as under `next dev`) is no
// aliases. Self-contained: app/layout.tsx serializes it into the page's head,
// and docs/test_redirects.jl runs that text alone.
export function anchorDestination(
  base: string,
  hash: string,
  path: string,
): Promise<AnchorDestination | undefined> {
  let id: string
  try {
    id = decodeURIComponent(hash.slice(1))
  } catch {
    return Promise.resolve(undefined) // a hash that does not decode names no anchor
  }
  if (
    id === '' ||
    document.getElementById(id) !== null ||
    !path.startsWith(`${base}/`)
  ) {
    return Promise.resolve(undefined)
  }
  let isRecord = (x: unknown): x is Record<string, unknown> =>
    typeof x === 'object' && x !== null && !Array.isArray(x)
  let isAliases = (x: unknown): x is Record<string, string> =>
    isRecord(x) && Object.values(x).every((to) => typeof to === 'string')
  let isAnchors = (x: unknown): x is Record<string, Array<string>> =>
    isRecord(x) &&
    Object.values(x).every(
      (routes) =>
        Array.isArray(routes) &&
        routes.length > 0 &&
        routes.every(
          (r) => typeof r === 'string' && /^\/([^/?#]+\/)*$/.test(r),
        ),
    )
  let read = <T>(
    name: string,
    valid: (x: unknown) => x is T,
    absent?: T,
  ): Promise<T> =>
    fetch(`${base}/${name}`)
      .then((response) => {
        if (response.status === 404 && absent !== undefined) return absent
        if (!response.ok) throw new Error(`HTTP ${response.status}`)
        return response.json().then((x: unknown) => {
          if (!valid(x)) throw new Error('not a map of the documented shape')
          return x
        })
      })
      .catch((error: Error) => {
        throw new Error(`${name}: ${error.message}`)
      })
  return Promise.all([
    read('anchor-aliases.json', isAliases, {}),
    read('anchors.json', isAnchors),
  ]).then(
    ([aliases, anchors]) => {
      let to = Object.hasOwn(aliases, id) ? aliases[id] : id
      let routes = Object.hasOwn(anchors, to) ? anchors[to] : []
      let rest = path.slice(base.length)
      let route = rest.endsWith('/') ? rest : `${rest}/`
      if (routes.includes(route)) {
        return to === id ? undefined : { route, id: to, here: true }
      }
      return routes.length === 1
        ? { route: routes[0], id: to, here: false }
        : undefined
    },
    (error: Error) => {
      console.error(`No anchor forwarding: ${error.message}`)
      return undefined
    },
  )
}
