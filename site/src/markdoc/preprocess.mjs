import Markdoc from '@markdoc/markdoc'
import yaml from 'js-yaml'
import { createLoader } from 'simple-functional-loader'

// the options the Markdoc loader tokenizes a page with
const tokenizer = new Markdoc.Tokenizer({ allowComments: true })
const LITERAL = '{% process=false %}'

// What the site changes in a generated page before Markdoc reads it:
// - every fence opts out of Markdoc processing. Markdoc reads `{% … %}` inside
//   fenced code, so code printing a template such as `{% for … %}` would open a
//   tag that swallows the rest of the page;
// - the <title> becomes "<title> - UniLM.jl", as an absolute title: Next.js does
//   not apply the layout's title template to the home page, which shares the
//   layout's route segment.
export function preprocess(source) {
  let lines = source.replace(/\r\n?/g, '\n').split('\n')
  let tokens = tokenizer.tokenize(lines.join('\n'))

  for (let token of tokens) {
    if (token.type === 'fence' && !token.info.includes(LITERAL)) {
      lines[token.map[0]] += ` ${LITERAL}`
    }
  }

  let frontmatter = tokens.find((token) => token.type === 'frontmatter')
  let data = frontmatter && yaml.load(frontmatter.content)
  let title = data?.nextjs?.metadata?.title
  if (typeof data?.title !== 'string' || typeof title !== 'string') {
    throw new Error(
      'a page needs a frontmatter with `title` and `nextjs.metadata.title` strings',
    )
  }
  data.nextjs.metadata.title = { absolute: `${title} - UniLM.jl` }

  // between the two `---` lines; padded to the original line count, so the
  // lines of Markdoc's errors still match the page
  let [open, close] = frontmatter.map
  let count = close - open - 1
  let body = yaml.dump(data, { flowLevel: 3, lineWidth: -1 }).trimEnd()
  let bodyLines = body.split('\n')
  let padding = Array(Math.max(0, count - bodyLines.length)).fill('')
  lines.splice(open + 1, count, ...bodyLines, ...padding)

  return lines.join('\n')
}

export default function withPreprocessedPages(nextConfig = {}) {
  return Object.assign({}, nextConfig, {
    webpack(config, options) {
      config.module.rules.push({
        test: /\.md$/,
        enforce: 'pre',
        use: [
          createLoader(function (source) {
            return preprocess(source)
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
