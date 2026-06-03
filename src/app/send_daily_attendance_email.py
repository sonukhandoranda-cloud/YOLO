from __future__ import annotations

import argparse
import os
import smtplib
from datetime import date
from email.message import EmailMessage
from pathlib import Path

from src.app.export_attendance_excel import export_attendance_excel


def required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing environment variable: {name}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and email the daily attendance Excel report.")
    parser.add_argument("--db", default="data/attendance/attendance.sqlite3")
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--session")
    parser.add_argument("--to", default=os.getenv("ATTENDANCE_REPORT_TO"))
    parser.add_argument("--subject", default=None)
    parser.add_argument("--output-dir", default="data/processed/attendance_exports")
    args = parser.parse_args()

    smtp_host = required_env("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = required_env("SMTP_USER")
    smtp_password = required_env("SMTP_PASSWORD")
    sender = os.getenv("SMTP_FROM", smtp_user)
    recipient = args.to
    if not recipient:
        raise RuntimeError("Provide --to or set ATTENDANCE_REPORT_TO.")

    report_path = export_attendance_excel(
        db=args.db,
        report_date=args.date,
        session=args.session,
        output_dir=args.output_dir,
    )

    subject = args.subject or f"Attendance Report - {args.date}"
    message = EmailMessage()
    message["From"] = sender
    message["To"] = recipient
    message["Subject"] = subject
    message.set_content(
        "Please find the attached attendance report.\n\n"
        "This email was generated automatically by the AI Attendance System."
    )

    path = Path(report_path)
    message.add_attachment(
        path.read_bytes(),
        maintype="application",
        subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=path.name,
    )

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(message)

    print(f"Sent attendance report to {recipient}")
    print(f"Attachment: {report_path}")


if __name__ == "__main__":
    main()
