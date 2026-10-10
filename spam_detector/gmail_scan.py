"""OPTIONAL: scan your recent Gmail messages with the trained model (read-only).

Setup:
  1. Go to https://console.cloud.google.com, create a project, enable the Gmail API.
  2. Create OAuth client credentials (type: Desktop app), download as credentials.json
     into this folder.
  3. pip install -r requirements.txt
  4. python gmail_scan.py --max 20

This only READS mail (gmail.readonly scope). It never deletes or moves anything.
"""
import argparse
import base64
import os
import joblib
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


def get_service():
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as f:
            f.write(creds.to_json())
    return build("gmail", "v1", credentials=creds)


def body_text(payload):
    if payload.get("mimeType", "").startswith("text/plain") and payload.get("body", {}).get("data"):
        return base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", "ignore")
    for part in payload.get("parts", []) or []:
        t = body_text(part)
        if t:
            return t
    return ""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=20)
    ap.add_argument("--query", default="in:inbox")
    args = ap.parse_args()

    model = joblib.load("models/spam_model.joblib")
    svc = get_service()
    msgs = svc.users().messages().list(userId="me", q=args.query, maxResults=args.max).execute().get("messages", [])
    for m in msgs:
        full = svc.users().messages().get(userId="me", id=m["id"], format="full").execute()
        headers = {h["name"]: h["value"] for h in full["payload"]["headers"]}
        subject = headers.get("Subject", "(no subject)")
        text = subject + " " + body_text(full["payload"])
        p = model.predict_proba([text])[0][1]
        tag = "SPAM" if p >= 0.5 else "ham "
        print(f"[{tag}] {p:6.1%}  {headers.get('From', '?')[:35]:35}  {subject[:60]}")
