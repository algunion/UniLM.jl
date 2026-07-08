# Doc-coverage ledger for the gate in docs/doc_coverage.jl.
#
# CURRENTLY EMPTY: every exported symbol is documented in an `@docs` block, so the
# gate now enforces FULL coverage — adding a new export without an `@docs` block
# fails the docs build. Only add a name here as a deliberate, temporary exception
# (with a reason in a comment), and remove it once the symbol is documented.
const KNOWN_UNDOCUMENTED = Set{String}([
])
