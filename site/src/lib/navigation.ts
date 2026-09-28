import data from '@/navigation.json'

export interface NavigationLink {
  title: string
  href: string
}

export interface NavigationGroup {
  title: string
  links: Array<NavigationLink>
}

// navigation.json is generated from docs/make.jl's `pages`; a malformed file
// fails the build here instead of rendering a broken sidebar.
function invalid(where: string, expected: string): never {
  throw new Error(`navigation.json: ${where} must be ${expected}`)
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function parseTitle(value: unknown, where: string) {
  return typeof value === 'string' && value !== ''
    ? value
    : invalid(where, 'a non-empty string')
}

function parseHref(value: unknown, where: string) {
  return typeof value === 'string' && /^\/([^/?#]+\/)*$/.test(value)
    ? value
    : invalid(where, 'a route such as "/" or "/guide/jev_start/"')
}

function parseGroup(value: unknown, where: string): NavigationGroup {
  if (!isRecord(value) || !Array.isArray(value.links)) {
    invalid(where, 'an object { "title": …, "links": […] }')
  }
  return {
    title: parseTitle(value.title, `${where}.title`),
    links: value.links.map((link: unknown, index) =>
      isRecord(link)
        ? {
            title: parseTitle(link.title, `${where}.links[${index}].title`),
            href: parseHref(link.href, `${where}.links[${index}].href`),
          }
        : invalid(`${where}.links[${index}]`, 'an object { "title", "href" }'),
    ),
  }
}

function parseNavigation(value: unknown): Array<NavigationGroup> {
  if (!Array.isArray(value)) {
    invalid('the top level', 'an array of groups')
  }
  let groups = value.map((group, index) => parseGroup(group, `[${index}]`))
  let hrefs = groups.flatMap((group) => group.links.map((link) => link.href))
  let repeated = hrefs.find((href, index) => hrefs.indexOf(href) !== index)
  return repeated === undefined
    ? groups
    : invalid(`route ${repeated}`, 'listed once')
}

export const navigation = parseNavigation(data)

// Every route ends in "/" (`trailingSlash: true`); usePathname() may return a
// route without it, so compare both in the trailing-slash form.
export function isCurrentRoute(href: string, pathname: string) {
  return href === (pathname.endsWith('/') ? pathname : `${pathname}/`)
}
