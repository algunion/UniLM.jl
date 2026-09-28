import { Callout } from '@/components/Callout'
import { Details } from '@/components/Details'
import { Docstring } from '@/components/Docstring'

const tags = {
  callout: {
    attributes: {
      title: { type: String },
      type: {
        type: String,
        default: 'note',
        matches: ['note', 'tip', 'warning'],
        errorLevel: 'critical',
      },
    },
    render: Callout,
  },
  details: {
    attributes: {
      summary: { type: String, required: true },
    },
    render: Details,
  },
  docstring: {
    attributes: {
      id: { type: String, required: true },
      name: { type: String, required: true },
      kind: { type: String, required: true },
    },
    render: Docstring,
  },
}

export default tags
