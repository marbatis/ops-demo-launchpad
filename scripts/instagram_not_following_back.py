#!/usr/bin/env python3
"""List Instagram accounts you follow that do not follow you back.

Works offline on Instagram's official data export ("Download your
information"), so no login, scraping, or third-party app is needed.

How to get the export:
  Instagram app -> Settings -> Accounts Center -> Your information and
  permissions -> Download your information -> Download or transfer
  information -> (pick the account) -> Some of your information ->
  "Followers and following" -> Download to device -> Format: JSON
  (HTML also works) -> Date range: All time -> Create files.

Usage:
  python3 scripts/instagram_not_following_back.py path/to/export.zip
  python3 scripts/instagram_not_following_back.py path/to/unzipped_folder \
      --csv not_following_back.csv

The path may be the downloaded .zip, the unzipped folder, or the
`followers_and_following` folder itself. Only Python's standard library
is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Iterable, Iterator

FOLLOWERS_RE = re.compile(r"(^|/)followers(_\d+)?\.(json|html)$")
FOLLOWING_RE = re.compile(r"(^|/)following\.(json|html)$")
PROFILE_HREF_RE = re.compile(
    r"https?://(?:www\.)?instagram\.com/(?:_u/)?([A-Za-z0-9._]+)/?", re.IGNORECASE
)


@dataclass(frozen=True)
class Account:
    username: str
    href: str
    followed_at: int | None = None  # unix timestamp from the export, if present

    @property
    def followed_on(self) -> str:
        if not self.followed_at:
            return ""
        return datetime.fromtimestamp(self.followed_at, tz=timezone.utc).date().isoformat()


def _normalize(username: str) -> str:
    return username.strip().lstrip("@").lower()


def _username_from_href(href: str) -> str | None:
    match = PROFILE_HREF_RE.match(href or "")
    return match.group(1) if match else None


def _iter_json_entries(payload: object) -> Iterator[dict]:
    """Yield relationship entries from either export layout.

    followers_N.json is a bare list; following.json is
    {"relationships_following": [...]}.
    """
    if isinstance(payload, list):
        yield from (e for e in payload if isinstance(e, dict))
    elif isinstance(payload, dict):
        for value in payload.values():
            if isinstance(value, list):
                yield from (e for e in value if isinstance(e, dict))


def parse_json(text: str) -> list[Account]:
    accounts: list[Account] = []
    for entry in _iter_json_entries(json.loads(text)):
        items = entry.get("string_list_data") or [{}]
        item = items[0] if isinstance(items[0], dict) else {}
        href = item.get("href", "")
        # Older exports put the username in "value"; newer ones put it in
        # the entry "title" and leave "value" out.
        username = item.get("value") or entry.get("title") or _username_from_href(href)
        if not username:
            continue
        accounts.append(
            Account(
                username=username,
                href=href or f"https://www.instagram.com/{username}",
                followed_at=item.get("timestamp"),
            )
        )
    return accounts


class _ProfileLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            href = dict(attrs).get("href") or ""
            if _username_from_href(href):
                self.hrefs.append(href)


def parse_html(text: str) -> list[Account]:
    parser = _ProfileLinkParser()
    parser.feed(text)
    seen: dict[str, Account] = {}
    for href in parser.hrefs:
        username = _username_from_href(href)
        if username and _normalize(username) not in seen:
            seen[_normalize(username)] = Account(username=username, href=href)
    return list(seen.values())


def _parse_file(name: str, text: str) -> list[Account]:
    return parse_json(text) if name.lower().endswith(".json") else parse_html(text)


def _prefer_json(names: Iterable[str]) -> list[str]:
    """If both JSON and HTML copies exist, keep only the JSON ones."""
    names = sorted(names)
    if any(n.endswith(".json") for n in names):
        return [n for n in names if n.endswith(".json")]
    return names


def load_export(path: Path) -> tuple[list[Account], list[Account]]:
    """Return (followers, following) read from a zip or a folder."""
    files: dict[str, str] = {}
    if path.is_file() and zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                posix = PurePosixPath(name).as_posix()
                if FOLLOWERS_RE.search(posix) or FOLLOWING_RE.search(posix):
                    files[posix] = zf.read(name).decode("utf-8")
    elif path.is_dir():
        for file in path.rglob("*"):
            posix = file.relative_to(path).as_posix()
            if file.is_file() and (FOLLOWERS_RE.search(posix) or FOLLOWING_RE.search(posix)):
                files[posix] = file.read_text(encoding="utf-8")
    else:
        raise SystemExit(f"Not a zip file or folder: {path}")

    follower_files = _prefer_json(n for n in files if FOLLOWERS_RE.search(n))
    following_files = _prefer_json(n for n in files if FOLLOWING_RE.search(n))
    if not follower_files or not following_files:
        raise SystemExit(
            "Could not find followers_*.json/html and following.json/html in the export. "
            "Make sure you requested 'Followers and following' in Download your information."
        )

    followers = [a for n in follower_files for a in _parse_file(n, files[n])]
    following = [a for n in following_files for a in _parse_file(n, files[n])]
    return followers, following


def not_following_back(followers: list[Account], following: list[Account]) -> list[Account]:
    follower_names = {_normalize(a.username) for a in followers}
    result: dict[str, Account] = {}
    for account in following:
        key = _normalize(account.username)
        if key not in follower_names and key not in result:
            result[key] = account
    return sorted(result.values(), key=lambda a: _normalize(a.username))


def write_csv(accounts: list[Account], destination: Path) -> None:
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["username", "profile_url", "followed_on"])
        for a in accounts:
            writer.writerow([a.username, a.href, a.followed_on])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("export", type=Path, help="Instagram export .zip or unzipped folder")
    parser.add_argument("--csv", type=Path, help="Also write the list to this CSV file")
    args = parser.parse_args(argv)

    followers, following = load_export(args.export)
    result = not_following_back(followers, following)

    print(f"Followers: {len({_normalize(a.username) for a in followers})}")
    print(f"Following: {len({_normalize(a.username) for a in following})}")
    print(f"Not following back: {len(result)}\n")
    for a in result:
        print(f"{a.username}\t{a.href}")

    if args.csv:
        write_csv(result, args.csv)
        print(f"\nSaved CSV to {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
