'use client'

import { Fragment } from 'react'
import { Highlight } from 'prism-react-renderer'

import { Output } from '@/components/Output'
import '@/lib/prism'

// A language Prism has no grammar for (`julia-repl`, `text`, …) renders as
// plain text.
export function Fence({
  children,
  language,
}: {
  children: string
  language?: string
}) {
  if (language === 'output') {
    return <Output>{children}</Output>
  }

  return (
    <Highlight
      code={children.trimEnd()}
      language={language ?? 'text'}
      theme={{ plain: {}, styles: [] }}
    >
      {({ className, style, tokens, getTokenProps }) => (
        <pre className={className} style={style}>
          <code>
            {tokens.map((line, lineIndex) => (
              <Fragment key={lineIndex}>
                {line
                  .filter((token) => !token.empty)
                  .map((token, tokenIndex) => (
                    <span key={tokenIndex} {...getTokenProps({ token })} />
                  ))}
                {'\n'}
              </Fragment>
            ))}
          </code>
        </pre>
      )}
    </Highlight>
  )
}
