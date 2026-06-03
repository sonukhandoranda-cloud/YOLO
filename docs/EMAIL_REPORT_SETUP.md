# Daily Email Report Setup

The system can export attendance to Excel and send it by email at the end of the day.

## Required Environment Variables

Do not store email passwords in GitHub.

Set these environment variables on the machine that sends the report:

```text
SMTP_HOST
SMTP_PORT
SMTP_USER
SMTP_PASSWORD
SMTP_FROM
ATTENDANCE_REPORT_TO
```

Example for Gmail:

```text
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password
SMTP_FROM=your_email@gmail.com
ATTENDANCE_REPORT_TO=owner_or_professor_email@example.com
```

For Gmail, use an app password, not your normal account password.

## Send Manually

```bash
python -m src.app.send_daily_attendance_email --date YYYY-MM-DD
```

Optional session filter:

```bash
python -m src.app.send_daily_attendance_email --date YYYY-MM-DD --session face-demo
```

## Automate End Of Day

On Windows, use Task Scheduler:

```text
Trigger: Daily at end-of-day time
Action: python -m src.app.send_daily_attendance_email
Start in: C:\Users\BIT\Desktop\Attendence
```

On Raspberry Pi/Linux, use cron:

```text
55 17 * * * cd /path/to/Attendence && python -m src.app.send_daily_attendance_email
```

## Output

The email contains the generated Excel file:

```text
data/processed/attendance_exports/
```
