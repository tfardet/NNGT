====================
Contributing to NNGT
====================


.. contents::
   :local:


Signaling issues and bugs
=========================

If you encounter something that you think is an error, please let me know
either via the `user mailing list <https://lists.sr.ht/~tfardet/nngt-users>`_
or directly on the `issue tracker <https://github.com/tfardet/NNGT/issues>`_.

.. warning ::
    When signaling a bug, please **always** include a python script containing
    a minimal working example (MWE) that reproduces the issue.


Preparing a contribution
========================

To prepare a contribution to NNGT, you should follow these successive steps:

1. start from the ``main`` branch: ``git checkout main``,
2. create a new branch from ``main``: ``git checkout -b name-of-your-choice``,
3. make the changes you want to and commit them,
4. check them locally using: ``pytest testing`` (you'll need to install pytest
   via ``pip install pytest``)


Sending a patch to SourceHut
============================

To contribute on SourceHut, you don't need an account (though you can also
make a patch using the website if you have an account there).

What you need is to use ``git send-email``, and you can find how to install and
set it up on `this page <https://git-send-email.io>`_.

Before sending you patch, please squash you commits using: ::

    git checkout -b patch-branch
    git merge --squash name-of-your-choice
    git checkout -a -m "A descriptive message of the changes"


First contribution
------------------

Once this is done, you can push your patch to the mailing list using: ::

    git send-email --annotate --to=~tfardet/nngt-developers@lists.sr.ht -v1 HEAD^

you can add further information in the description using annotate.

.. warning::
    Always use ``--annotate`` because you will need to change the subject from
    "[PATCH v1]" to "[PATCH NNGT]" or "[PATCH NNGT v1]" (as you prefer as long
    as the second word is NNGT) so that the patch is automatically tested on
    SourceHut

Do not hesitate to ask for help on the `developer mailing list
<https://lists.sr.ht/~tfardet/nngt-developers>`_ if you need help
on your first contribution.


Post-review changes: later contributions
----------------------------------------

If changes are requested, apply the changes to the branch
``name-of-your-choice``, then reset ``patch-branch`` ::

    git checkout patch-branch
    git fetch origin
    git reset --hard origin/main
    git merge --squash name-of-your-choice
    git checkout -a -m "A descriptive message of the changes"

then, publish the patch saying it's a new version: ::

    git send-email --annotate --to=~tfardet/nngt-developers@lists.sr.ht -v2 HEAD^

Or -v3, -v4, etc for later patches.

.. warning::
    As before, use annotate to change the subject to "[PATCH NNGT]" or
    "[PATCH NNGT v2]" so that the patch is automatically tested on SourceHut


Making a PR on GitHub
=====================

If you prefer using GitHub, then you can
`open a PR on the repo <https://github.com/tfardet/NNGT/pulls>`_.
