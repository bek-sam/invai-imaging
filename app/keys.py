"""Storage-key scope checks for the listing-photo routes (T-26-2).

Keys are `<companyId>/<kind>/...` (the backend's `objectKey`). The backend already checks every
key it sends against the caller's company (`isCompanyKey`); imaging checks again so one call can
never read one company's files and write them under another's prefix: every input key must be
well formed and share `out_key`'s company prefix. A violation is a 400 (KeyScopeError).
"""

import re

# Same shape as the backend's isSafeKey (src/lib/s3.ts): no empty segments, no `.`/`..`, no
# leading slash, 512 chars max.
SAFE_KEY = re.compile(r"^[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-][A-Za-z0-9._-]*)*$")
COMPANY_ID = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


class KeyScopeError(Exception):
    """A storage key that is malformed or outside the call's company prefix (HTTP 400)."""


def company_of(key: str, what: str = "key") -> str:
    """The company id a key lives under, or KeyScopeError."""
    if len(key) > 512 or not SAFE_KEY.match(key) or ".." in key.split("/"):
        raise KeyScopeError(f"{what} is not a valid storage key")
    parts = key.split("/")
    if len(parts) < 2 or not COMPANY_ID.match(parts[0]):
        raise KeyScopeError(f"{what} is not under a company prefix")
    return parts[0]


def check_scope(out_key: str, inputs: dict[str, str]) -> str:
    """Every input key must sit under `out_key`'s company prefix. Returns that company id."""
    company = company_of(out_key, "out_key")
    for what, key in inputs.items():
        if company_of(key, what) != company:
            raise KeyScopeError(f"{what} is outside out_key's company prefix")
    return company
