# UniLM.jl documentation site

The Next.js site that serves the UniLM.jl manual, such as
https://algunion.github.io/UniLM.jl/dev/. Documenter still builds the manual:
it runs every example, expands `@docs`, resolves `@ref` and checks that every
export is documented. `docs/make.jl` then writes each processed page as Markdoc
into this directory, and the site renders it.

## Build

1. From the repository root, generate the pages:

   ```bash
   julia --project=docs docs/make.jl
   ```

   This writes `src/app/**/page.md`, `src/navigation.json` and
   `public/assets/`, all gitignored.

2. Build the static site:

   ```bash
   cd site
   npm ci
   DOCS_BASE_PATH=/UniLM.jl/dev npm run build
   ```

   The export is written to `out/`. `DOCS_BASE_PATH` is the path the site is
   served under (unset for a domain root); pages never contain it.
   `NEXT_PUBLIC_DOCS_VERSION` sets the version label in the header: `dev` (the
   default), `stable` or a release such as `v0.22.0`.

`npm run fixtures` replaces the generated pages with the three sample pages in
`fixtures/`, written in the writer's output format, so the site can be built
and reviewed without generating the manual. `npm run dev` serves the site with
live reload; `npm run lint` and `npx tsc --noEmit` check the code.

## Licence

The site is built on Syntax, a Tailwind Plus template, under the Tailwind Plus
licence held by the maintainer (`LICENSE.md`). `site/` is not covered by the
repository's MIT licence, and it is not a template, theme or starter kit: it
must not be reused or redistributed as one.
