import importlib.util
import json
import sys
import zipfile
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "instagram_not_following_back",
    Path(__file__).resolve().parents[1] / "scripts" / "instagram_not_following_back.py",
)
ig = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = ig
_SPEC.loader.exec_module(ig)

BASE = "connections/followers_and_following"

FOLLOWERS_1 = [
    {"title": "", "string_list_data": [{"href": "https://www.instagram.com/alice", "value": "alice", "timestamp": 1700000000}]},
    {"title": "", "string_list_data": [{"href": "https://www.instagram.com/Bob", "value": "Bob", "timestamp": 1700000001}]},
]
FOLLOWERS_2 = [
    {"title": "", "string_list_data": [{"href": "https://www.instagram.com/carol", "value": "carol", "timestamp": 1700000002}]},
]
# Newer export layout: username in "title", no "value".
FOLLOWING = {
    "relationships_following": [
        {"title": "alice", "string_list_data": [{"href": "https://www.instagram.com/_u/alice", "timestamp": 1710000000}]},
        {"title": "bob", "string_list_data": [{"href": "https://www.instagram.com/_u/bob", "timestamp": 1710000001}]},
        {"title": "carol", "string_list_data": [{"href": "https://www.instagram.com/_u/carol", "timestamp": 1710000002}]},
        {"title": "zed", "string_list_data": [{"href": "https://www.instagram.com/_u/zed", "timestamp": 1710000003}]},
        {"title": "dave", "string_list_data": [{"href": "https://www.instagram.com/_u/dave", "timestamp": 1710000004}]},
    ]
}


def _write_folder(root: Path) -> Path:
    folder = root / BASE
    folder.mkdir(parents=True)
    (folder / "followers_1.json").write_text(json.dumps(FOLLOWERS_1))
    (folder / "followers_2.json").write_text(json.dumps(FOLLOWERS_2))
    (folder / "following.json").write_text(json.dumps(FOLLOWING))
    return root


def test_folder_export_finds_non_followers_across_paged_files(tmp_path: Path) -> None:
    followers, following = ig.load_export(_write_folder(tmp_path))
    result = ig.not_following_back(followers, following)
    assert [a.username for a in result] == ["dave", "zed"]
    assert result[0].href == "https://www.instagram.com/_u/dave"
    assert result[0].followed_on == "2024-03-09"


def test_zip_export(tmp_path: Path) -> None:
    archive = tmp_path / "export.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(f"{BASE}/followers_1.json", json.dumps(FOLLOWERS_1 + FOLLOWERS_2))
        zf.writestr(f"{BASE}/following.json", json.dumps(FOLLOWING))
    followers, following = ig.load_export(archive)
    assert [a.username for a in ig.not_following_back(followers, following)] == ["dave", "zed"]


def test_html_export(tmp_path: Path) -> None:
    folder = tmp_path / BASE
    folder.mkdir(parents=True)
    (folder / "followers_1.html").write_text(
        '<div><a target="_blank" href="https://www.instagram.com/alice">alice</a></div>'
    )
    (folder / "following.html").write_text(
        '<a href="https://www.instagram.com/alice">alice</a>'
        '<a href="https://www.instagram.com/_u/zed">zed</a>'
        '<a href="https://help.instagram.com/">help</a>'
    )
    followers, following = ig.load_export(tmp_path)
    assert [a.username for a in ig.not_following_back(followers, following)] == ["zed"]


def test_cli_writes_csv(tmp_path: Path, capsys) -> None:
    out = tmp_path / "result.csv"
    assert ig.main([str(_write_folder(tmp_path / "export")), "--csv", str(out)]) == 0
    assert "Not following back: 2" in capsys.readouterr().out
    lines = out.read_text().splitlines()
    assert lines[0] == "username,profile_url,followed_on"
    assert lines[1].startswith("dave,")
