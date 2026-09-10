from __future__ import annotations

import typing

type NestedSequence[T] = T | typing.Sequence["NestedSequence[T]"]
type MaybeBlocked[T] = T | typing.Sequence[T]
type MaybeBlockedMatrix[T] = T | typing.Sequence[typing.Sequence[T]]
