"""Server-owned runtime boundaries, independent of browser/UI state."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimePolicy:
    mode: str = "public_demo"

    def __post_init__(self):
        if self.mode not in {"public_demo", "local_private"}:
            raise ValueError("APP_MODE must be public_demo or local_private")

    @property
    def is_public(self):
        return self.mode == "public_demo"

    def require_local_mutation(self):
        if self.is_public:
            raise PermissionError(
                "Uploads and persistent memory require authorized access; "
                "they are unavailable in public_demo mode."
            )

    def validate_launch(self, *, share=False, hosted=False):
        if not self.is_public and (share or hosted):
            raise ValueError(
                "local_private is single-user and loopback-only; "
                "use public_demo for hosted or shared deployments."
            )

    def scope_collection(self, collection):
        if not self.is_public or isinstance(collection, PublicCollection):
            return collection
        return PublicCollection(collection)


class PublicCollection:
    """Read-only Chroma view: unclassified and private chunks are inaccessible.

    Both query() and get() enforce the filter so lexical retrieval cannot
    bypass the dense-search boundary. No unrestricted attribute delegation or
    mutation methods are exposed.
    """

    access_scope = "public"

    def __init__(self, collection):
        self._collection = collection

    @property
    def name(self):
        return self._collection.name

    @property
    def id(self):
        return self._collection.id

    @staticmethod
    def _filters(kwargs):
        kwargs = dict(kwargs)
        public = {"visibility": "public"}
        where = kwargs.pop("where", None)
        kwargs["where"] = {"$and": [public, where]} if where else public
        return kwargs

    def get(self, **kwargs):
        return self._collection.get(**self._filters(kwargs))

    def query(self, **kwargs):
        return self._collection.query(**self._filters(kwargs))

    def count(self):
        # Chroma count() has no metadata filter. Fetch IDs only, never private
        # documents, for the modest showcase corpus.
        return len(self.get(include=[])["ids"])
