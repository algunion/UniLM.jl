// Puts site/fixtures where docs/make.jl writes the manual's pages, replacing
// whatever is there, so the site builds before the manual has been generated.
import glob from 'fast-glob'
import { cp, readdir, rm, rmdir } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const site = join(dirname(fileURLToPath(import.meta.url)), '..')
const app = join(site, 'src/app')

// Remove the generated pages, then the route directories they leave empty.
async function removeEmptyDirs(dir) {
  for (let entry of await readdir(dir, { withFileTypes: true })) {
    if (entry.isDirectory()) {
      await removeEmptyDirs(join(dir, entry.name))
    }
  }
  if (dir !== app && (await readdir(dir)).length === 0) {
    await rmdir(dir)
  }
}

for (let page of await glob('**/page.md', { cwd: app })) {
  await rm(join(app, page))
}
await removeEmptyDirs(app)
await rm(join(site, 'public/assets'), { recursive: true, force: true })

await cp(join(site, 'fixtures/app'), app, { recursive: true })
await cp(
  join(site, 'fixtures/navigation.json'),
  join(site, 'src/navigation.json'),
)
await cp(join(site, 'fixtures/assets'), join(site, 'public/assets'), {
  recursive: true,
})
