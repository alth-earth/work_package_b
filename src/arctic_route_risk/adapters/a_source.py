"""Document the only accepted public A live and persistent resolver paths."""

from __future__ import annotations

from typing import Protocol

from arctic_route_data import DatasetBundle, PreparedWindow


class PublicABundleResolver(Protocol):
    """Structural view of A's public exact-bundle resolver.

    ``WorkPackageA.resolve_dataset_bundle_for_b`` implements this protocol. B
    never scans A's manifest database, cache directories, or archive paths.
    """

    def resolve_dataset_bundle_for_b(
        self,
        bundle: DatasetBundle,
        *,
        generation_id: int,
        knowledge_as_of,
    ) -> PreparedWindow: ...


__all__ = ["PublicABundleResolver"]
