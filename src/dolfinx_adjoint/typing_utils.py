from __future__ import annotations

import typing

type NestedSequence[T] = T | typing.Sequence["NestedSequence[T]"]
type NestedMutableSequence[T] = T | typing.MutableSequence["NestedMutableSequence[T]"]
