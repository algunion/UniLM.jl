// <base>/anchor-aliases.json and <base>/anchors.json, written by the docs build
type Aliases = Record<string, string>
type Anchors = Record<string, Array<string>>

// Where a reader who opened this page with an anchor id it does not define
// belongs. A retired id becomes its successor (anchor-aliases.json; no aliases
// when it cannot be fetched), then the routes that define the id (anchors.json)
// decide: this page when it is one of them (`here`, the hash then naming the
// successor), else the one route that defines it. Resolves to undefined when
// the reader stays: the hash is empty, does not decode or names an element
// here, or several routes or none define the id. Self-contained: the script in
// app/layout.tsx serializes it.
export function anchorDestination(
  base: string,
): Promise<{ route: string; id: string; here: boolean } | undefined> {
  let id: string
  try {
    id = decodeURIComponent(location.hash.slice(1))
  } catch {
    return Promise.resolve(undefined) // a hash that does not decode names no anchor
  }
  if (id === '' || document.getElementById(id) !== null) {
    return Promise.resolve(undefined)
  }
  let read = (name: string) =>
    fetch(`${base}/${name}`).then((response) => {
      if (!response.ok) throw new Error(`${name}: HTTP ${response.status}`)
      return response.json()
    })
  return Promise.all([
    read('anchor-aliases.json').catch(() => ({})),
    read('anchors.json'),
  ]).then(([aliases, anchors]: [Aliases, Anchors]) => {
    let to = Object.hasOwn(aliases, id) ? aliases[id] : id
    let routes = Object.hasOwn(anchors, to) ? anchors[to] : []
    let path = location.pathname.slice(base.length)
    let route = path.endsWith('/') ? path : `${path}/`
    if (routes.includes(route)) {
      return to === id ? undefined : { route, id: to, here: true }
    }
    return routes.length === 1
      ? { route: routes[0], id: to, here: false }
      : undefined
  })
}
