// Prism's language files register themselves on a global `Prism`: point it at
// the instance prism-react-renderer highlights with, then add the languages that
// instance does not bundle (it has JSON). The import order matters.
import '@/lib/prism-global'
import 'prismjs/components/prism-bash'
import 'prismjs/components/prism-julia'
import 'prismjs/components/prism-toml'
