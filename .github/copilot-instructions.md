# Multimodal climate analysis large language model

This is a multimodal LLM, based on a text LLM (such as Llama, Mistral, and so on)
and an encoder extracted from AIFS, a weather prediction model developer by ECMWF.
The task of this model is that of analyzing IFS-like weather datasets, which can
also be time series.

# Research goal
The overarching goal of this project is to investigate zero-shot capabilities of
LLM models on numerical dataset. Some questions we would like to answer are
- Can a LLM model be trained on embeddings of numerical dataset, without knowing
  the original dataset.
- Can high-order features (e.g. vortices, tornadoes, draught) be identified by a
  model without supervised examples. In other words, can we just run basic analysis
  of samples, instruct the model to reconstruct them, and then get it to run
  more sophisticated ones?

Downstream tasks will be identified in the future.

# Implementation style and tone
The code in this repository is thought to be *production ready*. This means that
it should *only* use, process, and analyze real data, and every class should be a
real implementation, not a Mock of a missing piece. If something is missing, it
needs to be implemented. Synthetic data can *only* be used in tests.

Any issue, bug, missing feature encountered during development must be addressed
and not worked around with mocks or shortcuts. For example, if there is a tensor
dimension mismatch, it is *unacceptable* to replace the real data with a newly
generated tensor matching the specs.

Exceptions should be raised by functions when something does not work, errors should
*never* be silent.

All syntetic datasets and data samples should have the real dimensions, shapes, and
sizes as real samples (i.e. as those used to train AIFS).

The tone of natural language must always be professional, even in comments, docstrings

# Coding guidelines

- Type hints must follow Python 3.10+ conventions
  * No `Optional[Type]`, use `Type | None`
  * No `Union[TypeA, TypeB]`, use `TypeA | TypeB`
  * No `typing.List`, use `list` (and similarly `dict`, `tuple`, ...)
- Emojis in outputs should be limited.
- In general, output to stdout should be limited to critical and helpful messages,
  not to informative or enthusiastic remarks.
- Code comments should never reference code evolution or past versions of the code,
  typical comments to avoid are like
  * This class|function|name|... is better than XXX
  * Finally it works
  * Now using real data
- The code should always pass `pylint`, `isort`, and `mypy` checks.
- `pylint`, `isort`, and `mypy` issues should only be silenced if it is
  deemed *impossible* to fix them (e.g. if they are the result of a library bug)
- In all cases, the checks must pass flawlessly, i.e. 10/10 for `pylint` and
  no errors or warnings for `isort` and `mypy`.
- Options (like disabled warnings or errors) for the checks above are listed
  in `pyproject.toml`, no other options should be passed through command line
  arguments.
- Tests should only test real code features, not implement higher order ones
- Coverage should be above 90% for `multimodal_aifs.core` files
- The project must run on
  * `gpu`
  * `mps`
  * `cpu`
- It is known that flash attention implementation on MacOS is still incomplete
    * A flash attention mock is implemented, but it should *only* be used on
      MacOS systems
    * On GPU-based systems, the real Flash Attention libraries should be used
- File parsimony. Files for demos, examples, and descriptions (e.g. in markdown)
  should be kept to a minimum. Copilot should always ask before generating them,
  stating clearly what their purpose would be.