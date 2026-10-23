import data from '@/navigation.json'

export interface NavigationLink {
  title: string
  href: string
}

export interface NavigationGroup {
  title: string
  links: Array<NavigationLink>
}

export interface NavigationFamily {
  title: string
  groups: Array<NavigationGroup>
}

export type NavigationItem = NavigationGroup | NavigationFamily

export function isFamily(item: NavigationItem): item is NavigationFamily {
  return 'groups' in item
}

// navigation.json is generated from docs/make.jl's `pages`; a malformed file
// fails the build here instead of rendering a broken sidebar.
function invalid(where: string, expected: string): never {
  throw new Error(`navigation.json: ${where} must be ${expected}`)
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

// The index of the first key equal to an earlier one, or -1.
function repeatIndex(keys: Array<string>) {
  return keys.findIndex((key, index) => keys.indexOf(key) !== index)
}

// Titles key the sidebar's items, and so a group's open state, and name a
// page's trail: siblings' titles differ.
function uniqueTitles<T extends { title: string }>(
  items: Array<T>,
  where: string,
  within: string,
) {
  let index = repeatIndex(items.map((item) => item.title))
  return index === -1
    ? items
    : invalid(`${where}[${index}].title`, `unique within ${within}`)
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
  if (!isRecord(value) || !Array.isArray(value.links) || 'groups' in value) {
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

// A family holds groups only.
function parseFamily(
  value: Record<string, unknown>,
  where: string,
): NavigationFamily {
  if (!Array.isArray(value.groups)) {
    invalid(`${where}.groups`, 'an array of groups')
  }
  return {
    title: parseTitle(value.title, `${where}.title`),
    groups: uniqueTitles(
      value.groups.map((group: unknown, index) =>
        parseGroup(group, `${where}.groups[${index}]`),
      ),
      `${where}.groups`,
      where,
    ),
  }
}

// An item with "groups" is a family, one with "links" a group; one with both
// or neither is refused.
function parseItem(value: unknown, where: string): NavigationItem {
  if (!isRecord(value) || 'links' in value === 'groups' in value) {
    invalid(
      where,
      'either a group { "title": …, "links": […] } or a family { "title": …, "groups": […] }',
    )
  }
  return 'groups' in value
    ? parseFamily(value, where)
    : parseGroup(value, where)
}

function parseNavigation(value: unknown): Array<NavigationItem> {
  if (!Array.isArray(value)) {
    invalid('the top level', 'an array of groups and families')
  }
  return uniqueTitles(
    value.map((item, index) => parseItem(item, `[${index}]`)),
    '',
    'the top level',
  )
}

function listedOnce(links: Array<NavigationLink>) {
  let index = repeatIndex(links.map((link) => link.href))
  return index === -1
    ? links
    : invalid(`route ${links[index].href}`, 'listed once')
}

export const navigation = parseNavigation(data)

function groupsOf(item: NavigationItem) {
  return isFamily(item) ? item.groups : [item]
}

// Every page in the sidebar's order, which previous and next follow.
export const allLinks = listedOnce(
  navigation.flatMap(groupsOf).flatMap((group) => group.links),
)

// Every route ends in "/" (`trailingSlash: true`); usePathname() may return a
// route without it, so compare both in the trailing-slash form.
export function isCurrentRoute(href: string, pathname: string) {
  return href === (pathname.endsWith('/') ? pathname : `${pathname}/`)
}

// Where the sidebar lists a route: the family (none for a top-level group) and
// the group, as a page's eyebrow and a search result's breadcrumb show them.
export function trailOf(
  pathname: string,
): { family?: string; group: string } | undefined {
  let holdsRoute = (group: NavigationGroup) =>
    group.links.some((link) => isCurrentRoute(link.href, pathname))
  for (let item of navigation) {
    let group = groupsOf(item).find(holdsRoute)
    if (group !== undefined) {
      return isFamily(item)
        ? { family: item.title, group: group.title }
        : { group: group.title }
    }
  }
}
