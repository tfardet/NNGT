"""
Sphinx extension taken from the Python doc-specific markup.
(https://github.com/python/cpython/blob/master/Doc/tools/extensions/pyspecific.py)

:copyright: 2008-2014 by Georg Brandl.
:license: Python license.
"""

from docutils import nodes, utils
from sphinx.util.nodes import split_explicit_title


SOURCE_URI = 'https://git.sr.ht/~tfardet/NNGT/tree/main/item/{}'


def source_role(typ, rawtext, text, lineno, inliner, options=None, content=None):
    '''
    Support for linking to Python source files easily through the :source:
    keyword
    '''
    options = {} if options is None else options
    options = {} if options is None else options

    _, title, target = split_explicit_title(text)

    title   = utils.unescape(title)
    target  = utils.unescape(target)
    refnode = nodes.reference(title, title,
                              refuri=SOURCE_URI.format(target))

    return [refnode], []


def setup(app):
    ''' Tell sphinx about the new function '''
    app.add_role('source', source_role)

    return {'version': '1.0', 'parallel_read_safe': True}
