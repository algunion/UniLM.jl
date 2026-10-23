'use client'

import { useId, useState } from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import clsx from 'clsx'

import {
  isCurrentRoute,
  isFamily,
  navigation,
  type NavigationGroup,
  type NavigationLink,
} from '@/lib/navigation'

// Required, though possibly undefined, so that every list passes on the mobile
// menu's handler, which closes the menu.
type OnLinkClick = React.MouseEventHandler<HTMLAnchorElement> | undefined

function ChevronIcon(props: React.ComponentPropsWithoutRef<'svg'>) {
  return (
    <svg aria-hidden="true" viewBox="0 0 16 16" fill="none" {...props}>
      <path
        d="m6 4 4 4-4 4"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}

function Links({
  links,
  pathname,
  onLinkClick,
  ...props
}: React.ComponentPropsWithoutRef<'ul'> & {
  links: Array<NavigationLink>
  pathname: string
  onLinkClick: OnLinkClick
}) {
  return (
    <ul
      role="list"
      className="mt-2 space-y-2 border-l-2 border-slate-100 lg:mt-4 lg:space-y-4 lg:border-slate-200 dark:border-slate-800"
      {...props}
    >
      {links.map((link) => {
        let current = isCurrentRoute(link.href, pathname)
        return (
          <li key={link.href} className="relative">
            <Link
              href={link.href}
              onClick={onLinkClick}
              aria-current={current ? 'page' : undefined}
              className={clsx(
                'block w-full pl-3.5 before:pointer-events-none before:absolute before:top-1/2 before:-left-1 before:h-1.5 before:w-1.5 before:-translate-y-1/2 before:rounded-full',
                current
                  ? 'font-semibold text-sky-500 before:bg-sky-500'
                  : 'text-slate-500 before:hidden before:bg-slate-300 hover:text-slate-600 hover:before:block dark:text-slate-400 dark:before:bg-slate-700 dark:hover:text-slate-300',
              )}
            >
              {link.title}
            </Link>
          </li>
        )
      })}
    </ul>
  )
}

// A family's group: open while it holds the current page, until the reader
// toggles it. The sidebar is not remounted on client navigation, so React keeps
// the toggle per family and group title (the keys) from page to page.
function Group({
  group,
  pathname,
  onLinkClick,
}: {
  group: NavigationGroup
  pathname: string
  onLinkClick: OnLinkClick
}) {
  // Navigation is mounted twice (sidebar and mobile menu) and group titles
  // repeat across families, so the list's id comes from useId().
  let id = useId()
  let [toggle, setToggle] = useState<boolean>()
  let current = group.links.some((link) => isCurrentRoute(link.href, pathname))
  let open = toggle ?? current

  return (
    <li>
      <h3>
        <button
          type="button"
          aria-expanded={open}
          aria-controls={id}
          onClick={() => setToggle(!open)}
          className={clsx(
            'flex w-full items-center gap-x-2 text-left font-medium',
            current
              ? 'text-sky-500'
              : 'text-slate-700 hover:text-slate-900 dark:text-slate-300 dark:hover:text-white',
          )}
        >
          {group.title}
          <ChevronIcon
            className={clsx(
              'ml-auto h-4 w-4 flex-none stroke-current transition-transform',
              open && 'rotate-90',
            )}
          />
        </button>
      </h3>
      <Links
        id={id}
        hidden={!open}
        links={group.links}
        pathname={pathname}
        onLinkClick={onLinkClick}
      />
    </li>
  )
}

export function Navigation({
  className,
  onLinkClick,
}: {
  className?: string
  onLinkClick?: React.MouseEventHandler<HTMLAnchorElement>
}) {
  let pathname = usePathname()

  return (
    <nav className={clsx('text-base lg:text-sm', className)}>
      <ul role="list" className="space-y-9">
        {navigation.map((item) => (
          <li key={item.title}>
            <h2 className="font-display font-medium text-slate-900 dark:text-white">
              {item.title}
            </h2>
            {isFamily(item) ? (
              <ul role="list" className="mt-2 space-y-2 lg:mt-4 lg:space-y-4">
                {item.groups.map((group) => (
                  <Group
                    key={group.title}
                    group={group}
                    pathname={pathname}
                    onLinkClick={onLinkClick}
                  />
                ))}
              </ul>
            ) : (
              <Links
                links={item.links}
                pathname={pathname}
                onLinkClick={onLinkClick}
              />
            )}
          </li>
        ))}
      </ul>
    </nav>
  )
}
