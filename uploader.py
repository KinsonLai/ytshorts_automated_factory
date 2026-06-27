import os
import json
import shutil
import glob
import time
import threading
from datetime import datetime
from typing import Any

import google_auth_oauthlib.flow
import googleapiclient.discovery
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google.auth.exceptions import RefreshError, TransportError

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
SESSION_FILE = "auth_sessions.json"


def _token_file(account_name: str) -> str:
    """Return the per-account token filename, e.g. ``youtube_token_MyChannel.json``."""
    if account_name and account_name != "Default":
        return f"youtube_token_{account_name}.json"
    return "youtube_token.json"


def _load_sessions() -> dict[str, Any]:
    """Load the auth session log from disk."""
    if os.path.exists(SESSION_FILE):
        try:
            with open(SESSION_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_sessions(sessions: dict[str, Any]) -> None:
    """Write the auth session log to disk."""
    try:
        with open(SESSION_FILE, "w", encoding="utf-8") as f:
            json.dump(sessions, f, indent=4)
    except Exception:
        pass


def _record_auth_event(account_name: str, event: str, success: bool = True) -> None:
    """Log a successful or failed OAuth event so the GUI can show status history."""
    sessions = _load_sessions()
    if account_name not in sessions:
        sessions[account_name] = {
            "last_auth": None, "last_refresh": None,
            "auth_count": 0, "refresh_count": 0,
            "last_error": None, "is_authenticated": False,
        }
    entry = sessions[account_name]
    entry[event] = datetime.now().isoformat()

    if event == "last_auth":
        entry["auth_count"] = entry.get("auth_count", 0) + 1
        entry["is_authenticated"] = success
    elif event == "last_refresh":
        entry["refresh_count"] = entry.get("refresh_count", 0) + 1
    elif event == "last_error":
        entry["is_authenticated"] = False

    _save_sessions(sessions)


def get_auth_status(account_name: str | None = None) -> dict[str, Any]:
    """Return a detailed snapshot of the current OAuth state for one account.

    Inspects the token file on disk plus the session journal to determine
    whether the account is ready for YouTube uploads and when the token expires.
    """
    if account_name is None:
        account_name = "Default"

    sessions = _load_sessions()
    entry = sessions.get(account_name, {})
    token_file = _token_file(account_name)

    status: dict[str, Any] = {
        "account": account_name,
        "is_authenticated": False,
        "token_exists": os.path.exists(token_file),
        "token_age_minutes": None,
        "expires_in_minutes": None,
        "last_auth": entry.get("last_auth"),
        "last_refresh": entry.get("last_refresh"),
        "refresh_count": entry.get("refresh_count", 0),
    }

    if os.path.exists(token_file):
        try:
            with open(token_file, "r", encoding="utf-8") as f:
                creds_data = json.loads(f.read())
            if creds_data.get("refresh_token"):
                status["has_refresh_token"] = True
                status["is_authenticated"] = True
            if creds_data.get("expiry"):
                expiry = datetime.fromisoformat(
                    creds_data["expiry"].replace("Z", "+00:00")
                )
                now = datetime.now(expiry.tzinfo)
                status["expires_at"] = expiry.isoformat()
                delta = expiry - now
                status["expires_in_minutes"] = int(delta.total_seconds() / 60)
                status["is_authenticated"] = delta.total_seconds() > 0
            mtime = os.path.getmtime(token_file)
            status["token_age_minutes"] = int((time.time() - mtime) / 60)
        except Exception:
            pass

    return status


def get_all_auth_statuses() -> dict[str, Any]:
    """Return auth status for every configured YouTube account at once."""
    accounts = list_available_accounts()
    result: dict[str, Any] = {"accounts": [], "total_authenticated": 0}
    for acc in accounts:
        s = get_auth_status(acc)
        result["accounts"].append(s)
        if s["is_authenticated"]:
            result["total_authenticated"] += 1
    return result


def _get_credentials(account_name: str) -> Credentials:
    """Obtain valid Google OAuth credentials for *account_name*.

    The priority order is:
    1. Load from an existing token file if it's still valid.
    2. Refresh the token if it's expired but a refresh token exists.
    3. Launch the browser-based OAuth flow and save a fresh token.
    """
    token_file = _token_file(account_name)
    credentials: Credentials | None = None

    if os.path.exists(token_file):
        try:
            with open(token_file, "r", encoding="utf-8") as f:
                credentials = Credentials.from_authorized_user_info(
                    json.loads(f.read()), SCOPES
                )
        except Exception:
            _record_auth_event(account_name, "last_error", success=False)

    if credentials and credentials.valid:
        return credentials

    if credentials and credentials.expired and credentials.refresh_token:
        try:
            with _refresh_lock():
                credentials.refresh(Request())
            with open(token_file, "w") as f:
                f.write(credentials.to_json())
            _record_auth_event(account_name, "last_refresh", success=True)
            print(f"[Tool: YouTube] Token auto-refreshed for {account_name}.")
            return credentials
        except (RefreshError, TransportError) as e:
            _record_auth_event(account_name, "last_error", success=False)
            print(f"[Tool: YouTube] Token refresh failed for {account_name}: {e}")
        except Exception:
            pass

    # need to run the interactive OAuth flow
    if account_name and account_name != "Default":
        secret_file = f"client_secret_{account_name}.json"
    else:
        secret_file = "client_secret.json"

    if not os.path.exists(secret_file):
        raise FileNotFoundError(f"Missing client_secret file: {secret_file}")

    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
    flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
        secret_file, SCOPES
    )
    credentials = flow.run_local_server(port=0)

    with open(token_file, "w") as f:
        f.write(credentials.to_json())
    _record_auth_event(account_name, "last_auth", success=True)
    print(f"[Tool: YouTube] New credentials saved for {account_name}.")
    return credentials


_refresh_lock_obj = threading.Lock()


def _refresh_lock() -> threading.Lock:
    """Return a lock that serialises token refresh to prevent race conditions."""
    return _refresh_lock_obj


def force_reauthenticate(account_name: str | None = None) -> bool:
    """Delete the stored token and re-run the OAuth flow from scratch."""
    if account_name is None:
        account_name = "Default"
    token_file = _token_file(account_name)
    if os.path.exists(token_file):
        os.remove(token_file)
    try:
        _get_credentials(account_name)
        return True
    except Exception:
        return False


def list_available_accounts() -> list[str]:
    """Return a list of account names based on ``client_secret_*.json`` files."""
    accounts: list[str] = []
    for path in glob.glob("client_secret*.json"):
        filename = os.path.basename(path)
        if filename == "client_secret.json":
            accounts.append("Default")
        else:
            name = filename[len("client_secret_"):-len(".json")]
            accounts.append(name)
    return accounts


def add_account(source_path: str, account_name: str) -> bool:
    """Copy a ``client_secret`` JSON into the project and name it."""
    dest = f"client_secret_{account_name}.json"
    try:
        shutil.copy2(source_path, dest)
        return True
    except (IOError, OSError):
        return False


def remove_account(account_name: str) -> bool:
    """Remove a YouTube account's secrets and token from disk.

    The ``Default`` account can't be removed.
    """
    if account_name == "Default":
        return False
    filepath = f"client_secret_{account_name}.json"
    token_file = _token_file(account_name)
    try:
        os.remove(filepath)
        if os.path.exists(token_file):
            os.remove(token_file)
        sessions = _load_sessions()
        if account_name in sessions:
            del sessions[account_name]
            _save_sessions(sessions)
        return True
    except (IOError, OSError):
        return False


def clear_token(account_name: str) -> bool:
    """Delete the stored OAuth token, forcing a re-auth on next use."""
    token_file = _token_file(account_name)
    try:
        if os.path.exists(token_file):
            os.remove(token_file)
            _record_auth_event(account_name, "last_error", success=False)
            return True
    except OSError:
        pass
    return False


def upload_to_youtube(
    video_file: str,
    title: str,
    description: str,
    account_name: str | None = None,
) -> tuple[str, str]:
    """Upload *video_file* as a public YouTube Short.

    Returns ``(video_id, url)``.
    """
    if account_name is None:
        from tools import _agent_settings
        account_name = (
            _agent_settings.get("selected_account", "Default")
            if _agent_settings
            else "Default"
        )

    print(f"[Tool: YouTube] Uploading to account: {account_name}")

    try:
        credentials = _get_credentials(account_name)
    except FileNotFoundError as e:
        print(f"[Tool: YouTube] {e}")
        raise

    youtube = googleapiclient.discovery.build(
        "youtube", "v3", credentials=credentials
    )

    request_body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": ["shorts", "trending", "viral"],
            "categoryId": "24",
        },
        "status": {
            "privacyStatus": "public",
            "selfDeclaredMadeForKids": False,
        },
    }

    media_file = MediaFileUpload(video_file, chunksize=-1, resumable=True)
    request = youtube.videos().insert(
        part="snippet,status", body=request_body, media_body=media_file
    )
    response = request.execute()
    video_id = response["id"]
    url = f"https://youtube.com/shorts/{video_id}"
    print(f"[Success] Video uploaded! URL: {url}")
    return video_id, url
