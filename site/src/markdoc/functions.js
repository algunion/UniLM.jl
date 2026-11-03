// @markdoc/next.js imports four schema modules from this directory for every page:
// config, tags, nodes and functions. It logs each one it cannot resolve and uses an
// empty object in its place, so this one is that empty object: the pages call no
// Markdoc function, and Markdoc's built-in ones stay available.
const functions = {}

export default functions
