"""The users_on_file directory: who a caller may claim to be, and their number.

Routes 1 and 2 of the contract read it. Session create checks a claim against
it, and the SMS code goes to the claimed user's `phone_e164`, inside the
detector. The full number leaves the process only in `GET /v1/users/{id}`,
which the agent calls to dial the callback.

Source, in order: `USERS_ON_FILE_JSON` (inline JSON, how the Modal secret
supplies it), then `USERS_ON_FILE_PATH` (a file). With neither, the directory
is empty and a loud warning is printed. A file that does not parse, or breaks
the schema, stops the process at startup.

Nothing here logs or raises a message that holds a name or a phone number:
validation errors name the field that failed, never its value.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

from callback_shared.contract import UserOnFile, UserSummary
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from detector.config import Settings

PHONE_E164 = re.compile(r"^\+[1-9]\d{7,14}$")

# Stripped from both ends of a name after whitespace, so "Eric Catalano." and
# "eric catalano?" find the same person. STT puts punctuation there.
_EDGE_PUNCTUATION = ".,!?"


class _FileUser(BaseModel):
    # Unknown fields are ignored, like every model on the wire, so a note
    # field in the real file is not a startup failure.
    model_config = ConfigDict(extra="ignore")

    user_id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)
    role: str
    phone_e164: str

    @field_validator("phone_e164")
    @classmethod
    def _e164(cls, value: str) -> str:
        if not PHONE_E164.match(value):
            # No value in the message: it would be a phone number.
            raise ValueError(r"must match ^\+[1-9]\d{7,14}$")
        return value


class _File(BaseModel):
    model_config = ConfigDict(extra="ignore")

    users: list[_FileUser]


class UsersFileError(ValueError):
    """The directory could not be loaded. Raised at startup, never per request."""


def normalise_name(raw: str) -> str:
    """Strip, strip edge punctuation, casefold, collapse inner whitespace."""

    return " ".join(raw.strip().strip(_EDGE_PUNCTUATION).casefold().split())


def _problems(error: ValidationError) -> str:
    # include_input=False, and only loc and msg, so a bad phone number or name
    # never lands in a traceback or a startup log.
    return "; ".join(
        f"{'.'.join(str(part) for part in e['loc']) or 'file'}: {e['msg']}"
        for e in error.errors(include_input=False)[:5]
    )


class UserDirectory:
    """In-memory, read-only after construction."""

    def __init__(self, users: list[UserOnFile], aliases: dict[str, list[str]]) -> None:
        seen: set[str] = set()
        for user in users:
            if user.user_id in seen:
                raise UsersFileError("users_on_file: user_id values must be unique")
            seen.add(user.user_id)
        self._users = list(users)
        self._by_id = {user.user_id: user for user in users}
        # Every normalised name a user answers to: the display name and each
        # alias. Built once, so a search is a scan of short strings.
        self._names: list[tuple[UserOnFile, frozenset[str]]] = [
            (
                user,
                frozenset(
                    name
                    for name in (
                        normalise_name(user.display_name),
                        *(normalise_name(a) for a in aliases.get(user.user_id, [])),
                    )
                    if name
                ),
            )
            for user in users
        ]

    # ------------------------------------------------------------ building

    @classmethod
    def from_json_text(cls, text: str) -> UserDirectory:
        try:
            doc = json.loads(text)
        except json.JSONDecodeError as exc:
            # The position, not the text around it.
            raise UsersFileError(
                f"users_on_file: not valid JSON (line {exc.lineno}, column {exc.colno})"
            ) from None
        try:
            parsed = _File.model_validate(doc)
        except ValidationError as exc:
            raise UsersFileError(f"users_on_file: {_problems(exc)}") from None
        users = [
            UserOnFile(
                user_id=u.user_id,
                display_name=u.display_name,
                role=u.role,
                phone_last_four=u.phone_e164[-4:],
                phone_e164=u.phone_e164,
            )
            for u in parsed.users
        ]
        return cls(users, {u.user_id: list(u.aliases) for u in parsed.users})

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        warn: Callable[[str], None] | None = None,
    ) -> UserDirectory:
        """Apply the source order above. Raises `UsersFileError` on a bad file."""

        say = warn or _print_warning
        if settings.users_on_file_json:
            return cls.from_json_text(settings.users_on_file_json)
        if settings.users_on_file_path:
            path = Path(settings.users_on_file_path)
            try:
                text = path.read_text(encoding="utf-8")
            except OSError as exc:
                raise UsersFileError(
                    f"users_on_file: USERS_ON_FILE_PATH cannot be read "
                    f"({type(exc).__name__})"
                ) from None
            return cls.from_json_text(text)
        say(
            "WARNING: neither USERS_ON_FILE_JSON nor USERS_ON_FILE_PATH is set, so "
            "the /v1 user directory is empty: every name search finds nobody and "
            "every POST /v1/sessions is a 404 user_not_found. Point "
            "USERS_ON_FILE_PATH at users_on_file.example.json for local runs."
        )
        return cls([], {})

    # ------------------------------------------------------------- reading

    def __len__(self) -> int:
        return len(self._users)

    def search(self, name: str) -> list[UserSummary]:
        """Exact match after normalisation, on the display name or an alias.

        In file order. The agent goes ahead only on exactly one match.
        """

        needle = normalise_name(name)
        if not needle:
            return []
        return [
            UserSummary(
                user_id=user.user_id,
                display_name=user.display_name,
                role=user.role,
                phone_last_four=user.phone_last_four,
            )
            for user, names in self._names
            if needle in names
        ]

    def get(self, user_id: str) -> UserOnFile | None:
        return self._by_id.get(user_id)


def _print_warning(message: str) -> None:
    # Loud on purpose, like the generated-token warning in detector/app.py:
    # the service is up but cannot identify anyone.
    print(message, flush=True)
