import Markdoc from '@markdoc/markdoc'
import { slugifyWithCounter } from '@sindresorhus/slugify'
import glob from 'fast-glob'
import * as fs from 'fs'
import yaml from 'js-yaml'
import * as path from 'path'
import { createLoader } from 'simple-functional-loader'
import * as url from 'url'

import { preprocess } from './preprocess.mjs'

const __filename = url.fileURLToPath(import.meta.url)
const slugify = slugifyWithCounter()

function toString(node) {
  let str =
    (node.type === 'text' || node.type === 'code') &&
    typeof node.attributes?.content === 'string'
      ? node.attributes.content
      : ''
  if ('children' in node) {
    for (let child of node.children) {
      str += toString(child)
    }
  }
  return str
}

// Collects [title, hash, content] entries into `sections` and returns the
// entry that text met next belongs to.
function extractSections(node, sections, current) {
  if (node.type === 'heading' && node.attributes.level <= 2) {
    let content = toString(node).trim()
    let entry = [content, node.attributes.id ?? slugify(content), []]
    sections.push(entry)
    return entry
  }
  if (node.type === 'heading' || node.type === 'paragraph') {
    current[2].push(toString(node).trim())
    return current
  }
  if (node.type === 'tag' && node.tag === 'docstring') {
    // found by its binding's name, linking to Documenter's anchor for it
    let entry = [node.attributes.name, node.attributes.id, []]
    sections.push(entry)
    node.children.reduce(
      (target, child) => extractSections(child, sections, target),
      entry,
    )
    return current
  }
  return node.children.reduce(
    (target, child) => extractSections(child, sections, target),
    current,
  )
}

export default function withSearch(nextConfig = {}) {
  let cache = new Map()

  return Object.assign({}, nextConfig, {
    webpack(config, options) {
      config.module.rules.push({
        test: __filename,
        use: [
          createLoader(function () {
            let pagesDir = path.resolve('./src/app')
            this.addContextDependency(pagesDir)

            let files = glob.sync('**/page.md', { cwd: pagesDir })
            let data = files.map((file) => {
              // routes end in "/", like the links in navigation.json
              let url =
                file === 'page.md'
                  ? '/'
                  : `/${file.replace(/\/page\.md$/, '')}/`
              let md = fs.readFileSync(path.join(pagesDir, file), 'utf8')

              let sections

              if (cache.get(file)?.[0] === md) {
                sections = cache.get(file)[1]
              } else {
                let ast = Markdoc.parse(preprocess(md))
                let title = yaml.load(ast.attributes.frontmatter ?? '')?.title
                if (typeof title !== 'string') {
                  throw new Error(`${file}: the frontmatter has no title`)
                }
                sections = [[title, null, []]]
                slugify.reset()
                extractSections(ast, sections, sections[0])
                cache.set(file, [md, sections])
              }

              return { url, sections }
            })

            // When this file is imported within the application
            // the following module is loaded:
            return `
              import FlexSearch from 'flexsearch'

              let sectionIndex = new FlexSearch.Document({
                tokenize: 'full',
                document: {
                  id: 'url',
                  index: 'content',
                  store: ['title', 'pageTitle'],
                },
                context: {
                  resolution: 9,
                  depth: 2,
                  bidirectional: true
                }
              })

              let data = ${JSON.stringify(data)}

              for (let { url, sections } of data) {
                for (let [title, hash, content] of sections) {
                  sectionIndex.add({
                    url: url + (hash ? ('#' + hash) : ''),
                    title,
                    content: [title, ...content].join('\\n'),
                    pageTitle: hash ? sections[0][0] : undefined,
                  })
                }
              }

              export function search(query, options = {}) {
                let result = sectionIndex.search(query, {
                  ...options,
                  enrich: true,
                })
                if (result.length === 0) {
                  return []
                }
                return result[0].result.map((item) => ({
                  url: item.id,
                  title: item.doc.title,
                  pageTitle: item.doc.pageTitle,
                }))
              }
            `
          }),
        ],
      })

      if (typeof nextConfig.webpack === 'function') {
        return nextConfig.webpack(config, options)
      }

      return config
    },
  })
}
