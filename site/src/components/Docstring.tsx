// One `@docs` binding. `id` is Documenter's anchor, the target of every
// cross-reference to the binding.
export function Docstring({
  id,
  name,
  kind,
  children,
}: {
  id: string
  name: string
  kind: string
  children: React.ReactNode
}) {
  return (
    <section
      id={id}
      className="group my-10 scroll-mt-28 rounded-2xl border border-slate-200 lg:scroll-mt-34 dark:border-slate-800"
    >
      <header className="not-prose flex flex-wrap items-center gap-x-3 gap-y-1 border-b border-slate-200 px-5 py-3 dark:border-slate-800">
        <code className="font-mono text-sm font-semibold break-all text-slate-900 dark:text-white">
          {name}
        </code>
        <span className="rounded-full bg-sky-50 px-2 py-0.5 font-display text-xs font-medium text-sky-700 ring-1 ring-sky-600/20 ring-inset dark:bg-sky-400/10 dark:text-sky-400 dark:ring-sky-400/30">
          {kind}
        </span>
        <a
          href={`#${id}`}
          aria-label={`Link to ${name}`}
          className="ml-auto font-mono text-sm text-slate-400 opacity-0 transition group-hover:opacity-100 hover:text-sky-500 focus:opacity-100"
        >
          #
        </a>
      </header>
      <div className="px-5 *:first:mt-4 *:last:mb-4">{children}</div>
    </section>
  )
}
