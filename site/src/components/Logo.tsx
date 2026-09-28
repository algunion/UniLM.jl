import clsx from 'clsx'

export function Logo({ className }: { className?: string }) {
  return (
    <span
      className={clsx(
        'font-display text-2xl font-medium tracking-tight text-slate-900 dark:text-white',
        className,
      )}
    >
      UniLM<span className="text-sky-500">.jl</span>
    </span>
  )
}

// A link of its own: it never sits inside the logo's home link.
export function PopperianMark({ className }: { className?: string }) {
  return (
    <a
      href="https://popperian.ai"
      className={clsx(
        'font-display text-xs whitespace-nowrap text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-300',
        className,
      )}
    >
      by popperian.ai
    </a>
  )
}
