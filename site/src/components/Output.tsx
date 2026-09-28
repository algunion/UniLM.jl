// What an example printed and returned, exactly as Documenter shows it: plain
// text on a light panel, wrapped, so it never reads as code to copy.
export function Output({ children }: { children: string }) {
  return (
    <figure className="not-prose my-6 overflow-hidden rounded-xl ring-1 ring-slate-200 dark:ring-slate-700">
      <figcaption className="border-b border-slate-200 bg-slate-50 px-4 py-1.5 font-display text-xs font-medium tracking-wide text-slate-500 uppercase dark:border-slate-700 dark:bg-slate-800/60 dark:text-slate-400">
        Output
      </figcaption>
      <pre className="bg-white px-4 py-3 font-mono text-sm leading-6 wrap-break-word whitespace-pre-wrap text-slate-700 dark:bg-slate-900 dark:text-slate-300">
        {children.replace(/\n$/, '')}
      </pre>
    </figure>
  )
}
