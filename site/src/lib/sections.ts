import { type Node } from '@markdoc/markdoc'
import { slugifyWithCounter } from '@sindresorhus/slugify'

interface HeadingNode extends Node {
  type: 'heading'
  attributes: {
    level: 1 | 2 | 3 | 4 | 5 | 6
    id?: string
    [key: string]: unknown
  }
}

type H2Node = HeadingNode & {
  attributes: {
    level: 2
  }
}

type H3Node = HeadingNode & {
  attributes: {
    level: 3
  }
}

function isHeadingNode(node: Node): node is HeadingNode {
  return (
    node.type === 'heading' &&
    [1, 2, 3, 4, 5, 6].includes(node.attributes.level) &&
    (typeof node.attributes.id === 'string' ||
      typeof node.attributes.id === 'undefined')
  )
}

function isH2Node(node: Node): node is H2Node {
  return isHeadingNode(node) && node.attributes.level === 2
}

function isH3Node(node: Node): node is H3Node {
  return isHeadingNode(node) && node.attributes.level === 3
}

// A heading's title includes its inline code spans.
function getNodeText(node: Node): string {
  return (node.children ?? [])
    .map((child) =>
      child.type === 'text' || child.type === 'code'
        ? String(child.attributes.content)
        : getNodeText(child),
    )
    .join('')
}

export type Subsection = H3Node['attributes'] & {
  id: string
  title: string
  children?: undefined
}

export type Section = H2Node['attributes'] & {
  id: string
  title: string
  children: Array<Subsection>
}

export function collectSections(
  nodes: Array<Node>,
  slugify = slugifyWithCounter(),
) {
  let sections: Array<Section> = []

  for (let node of nodes) {
    if (isH2Node(node) || isH3Node(node)) {
      let title = getNodeText(node)
      if (title) {
        // An explicit id (Documenter's anchor) is what links point to.
        let id = node.attributes.id ?? slugify(title)
        if (isH3Node(node)) {
          if (!sections[sections.length - 1]) {
            throw new Error(
              'Cannot add `h3` to table of contents without a preceding `h2`',
            )
          }
          sections[sections.length - 1].children.push({
            ...node.attributes,
            id,
            title,
          })
        } else {
          sections.push({ ...node.attributes, id, title, children: [] })
        }
      }
    }

    sections.push(...collectSections(node.children ?? [], slugify))
  }

  return sections
}
