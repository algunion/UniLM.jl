// A table wider than the column scrolls inside its own box instead of widening
// the page.
export function Table({ children }: { children: React.ReactNode }) {
  return (
    <div className="overflow-x-auto">
      <table>{children}</table>
    </div>
  )
}
