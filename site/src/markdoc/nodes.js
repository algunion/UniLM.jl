import Markdoc, { nodes as defaultNodes, Tag } from '@markdoc/markdoc'
import { slugifyWithCounter } from '@sindresorhus/slugify'
import yaml from 'js-yaml'

import { DocLink } from '@/components/DocLink'
import { DocsLayout } from '@/components/DocsLayout'
import { Fence } from '@/components/Fence'

let documentSlugifyMap = new Map()

// The Markdoc loader renders a page without validating it: an unknown tag, a
// bad attribute or a malformed `{% %}` annotation would vanish from the page
// without a trace. Fail the build instead.
function assertValid(document, config) {
  let errors = Markdoc.validate(document, config).filter(
    ({ error }) => error.level === 'error' || error.level === 'critical',
  )
  if (errors.length > 0) {
    let page = config.variables?.markdoc?.frontmatter?.title
    let messages = errors.map(
      ({ lines, error }) =>
        `  ${lines?.length ? `line ${lines[0] + 1}` : 'page'}: ${error.message}`,
    )
    throw new Error([`Invalid Markdoc in "${page}":`, ...messages].join('\n'))
  }
}

// Pages link and point images at routes without the base path.
function withBasePath(url) {
  return url.startsWith('/') && !url.startsWith('//')
    ? `${process.env.NEXT_PUBLIC_BASE_PATH}${url}`
    : url
}

const nodes = {
  document: {
    ...defaultNodes.document,
    render: DocsLayout,
    transform(node, config) {
      assertValid(node, config)
      documentSlugifyMap.set(config, slugifyWithCounter())

      return new Tag(
        this.render,
        {
          frontmatter: yaml.load(node.attributes.frontmatter),
          nodes: node.children,
        },
        node.transformChildren(config),
      )
    },
  },
  heading: {
    ...defaultNodes.heading,
    attributes: {
      ...defaultNodes.heading.attributes,
      // Documenter's anchors, verbatim: Markdoc's built-in `id` type rejects
      // ones that start with a digit or "@", such as "1.-Upload-the-file"
      id: { type: String },
    },
    transform(node, config) {
      let slugify = documentSlugifyMap.get(config)
      let attributes = node.transformAttributes(config)
      let children = node.transformChildren(config)
      let text = children.filter((child) => typeof child === 'string').join(' ')
      let id = attributes.id ?? slugify(text)

      return new Tag(
        `h${node.attributes.level}`,
        { ...attributes, id },
        children,
      )
    },
  },
  th: {
    ...defaultNodes.th,
    attributes: {
      ...defaultNodes.th.attributes,
      scope: {
        type: String,
        default: 'col',
      },
    },
  },
  fence: {
    render: Fence,
    attributes: defaultNodes.fence.attributes,
    // The code as written: a fence that opts out of processing (all of them,
    // see preprocess.mjs) has its text in `content`, not in children.
    transform(node) {
      return new Tag(this.render, { language: node.attributes.language }, [
        node.attributes.content,
      ])
    },
  },
  link: {
    ...defaultNodes.link,
    // badges are images wrapped in links
    children: [...defaultNodes.link.children, 'image'],
    render: DocLink,
  },
  image: {
    ...defaultNodes.image,
    transform(node, config) {
      let { src, alt, ...attributes } = node.transformAttributes(config)
      return new Tag('img', {
        ...attributes,
        src: withBasePath(src),
        // Markdoc keeps the alt text as written; CommonMark drops the
        // backslash of an escaped punctuation character
        alt: alt?.replace(/\\([!-/:-@[-`{-~])/g, '$1'),
      })
    },
  },
}

export default nodes
