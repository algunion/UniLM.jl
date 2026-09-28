// Prism's language files register themselves on a global `Prism`: point it at
// the instance prism-react-renderer highlights with, then add Julia, which that
// instance does not bundle. The import order matters.
import '@/lib/prism-global'
import 'prismjs/components/prism-julia'
