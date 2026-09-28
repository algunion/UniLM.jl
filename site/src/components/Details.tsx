// Documenter's `!!! details`: collapsed until the reader opens it.
export function Details({
  summary,
  children,
}: {
  summary: string
  children: React.ReactNode
}) {
  return (
    <details className="my-8 rounded-3xl bg-slate-50 px-6 py-4 dark:bg-slate-800/60 dark:ring-1 dark:ring-slate-300/10">
      <summary className="cursor-pointer font-display text-lg text-slate-900 marker:text-sky-500 dark:text-white">
        {summary}
      </summary>
      <div className="mt-4 *:first:mt-0 *:last:mb-0">{children}</div>
    </details>
  )
}
