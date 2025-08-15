# decision_trees

Decision trees for ArsMedicaTech decision tree tool

## Usage

What follows is the implementation code as used in [ArsMedicaTech](https://github.com/darren277/arsmedicatech) for deterministic decision tree lookups as an LLM tool.

```python
compare_ops: dict[str, Callable[[Any, Any], bool]] = {}

def register_compare_op(symbol: str, func: Callable[[Any, Any], bool]) -> None:
    """Allow the tree author to plug‑in new binary operators at runtime."""
    compare_ops[symbol] = func

register_compare_op('==', lambda x, y: x == y)
register_compare_op('!=', lambda x, y: x != y)
register_compare_op('>',  lambda x, y: x >  y)
register_compare_op('>=', lambda x, y: x >= y)
register_compare_op('<',  lambda x, y: x <  y)
register_compare_op('<=', lambda x, y: x <= y)
register_compare_op('in',     lambda x, y: x in y)
register_compare_op('not in', lambda x, y: x not in y)
register_compare_op('regex',  lambda x, pattern: re.fullmatch(pattern, str(x)) is not None)

BranchKey = Union[Tuple[str, Any], Callable[[Any], bool], Any]


def _choose_branch(
        branches: Dict[BranchKey, Any],
        arg: Any,
        subterm: str,
        path: list[str]
    ) -> Tuple[Any | None, list[str]]:
    """
    Choose a branch from the decision tree based on the provided argument.
    :param branches: A dictionary mapping branch keys to their target values.
    :param arg: The argument to match against the branches.
    :param subterm: A descriptive term for the argument, used in logging.
    :param path: A list to accumulate the logical path taken through the tree.
    :return: A tuple containing the matched key (or None if no match) and the updated path.
    """

    for key in branches:
        # a)  Callable predicate
        if callable(key):
            if key(arg):
                path.append(f"Checked {subterm}: predicate {key.__name__} → True")
                return key, path
        # b)  (‘<op>’, reference)
        elif isinstance(key, tuple) and len(key) == 2: # type: ignore
            op, ref = key # type: ignore
            if op not in compare_ops:
                raise ValueError(f"Unsupported operator {op!r}. Register it first.")
            if compare_ops[op](arg, ref):
                path.append(f"Checked {subterm}: {arg!r} {op} {ref!r}")
                return key, path # type: ignore
        # c)  Literal / Enum → equality
        else:
            if arg == key:
                path.append(f"Checked {subterm}: {arg!r} == {key!r}")
                return key, path # type: ignore
    # no match
    return None, path


def decision_tree_lookup(tree: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
    """
    Looks up a decision from a deterministic decision tree.

    :param tree: The decision tree structure, where each node is a dictionary with 'question' and 'branches'.
    :param kwargs: Keyword arguments representing the answers to the questions in the tree.
    :type kwargs: Any
    :return: A dictionary containing the final decision and the logical path taken.
    """
    path_taken: list[str] = []
    current_node: Dict[str, Any] = tree

    # Loop until we reach a final decision (a string)
    while 'question' in current_node and 'branches' in current_node:
        question = str(current_node.get('question', ''))
        branches = current_node['branches']
        branches: Dict[BranchKey, Any] = branches  # type: ignore
        matched = False

        for kw_name, value in kwargs.items():
            subterm = kw_name.replace('_', ' ') # crude NLP: map loan_purpose → "loan purpose"
            if subterm in question:
                key, path_taken = _choose_branch(branches, value, subterm, path_taken)
                if key is None:
                    return {"decision": "Error",
                            "reason": f"Invalid value for {subterm}: {value!r}",
                            "path_taken": path_taken}
                current_node = branches[key]
                matched = True
                break

        if not matched:
            return {"decision": "Error",
                    "reason": f"Question {question!r} could not be answered with supplied arguments.",
                    "path_taken": path_taken}

    # At this point, current_node is the final decision string: We've hit a leaf (a string)
    decision, *rest = str(current_node).split(' - ', 1)
    reason = rest[0] if rest else "No specific reason provided."
    return {
        "decision": decision,
        "reason": reason,
        "path_taken": path_taken
    }
```

## Extraction

Windows PowerShell:
1. Enter the following: `[System.Convert]::ToBase64String([System.IO.File]::ReadAllBytes("extraction/on-create.sh"))`
2. Paste output of that into `.env` as `ON_CREATE`.

Linux:
1. Enter the following: `base64 extraction/on-create.sh`
2. Paste output of that into `.env` as `ON_CREATE`.
