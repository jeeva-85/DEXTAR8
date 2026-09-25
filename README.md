# Credentials Directory

This directory is intended for local authentication files and cloud service account credentials when deploying or connecting to external services (e.g., GCP, Firebase, or external railway APIs).

## Security Rules
1. **NEVER commit private keys or real credentials to GitHub.**
2. All credential files (`service-account.json`, `credentials.json`, `*.pem`, `*.key`, `*.secret`) are strictly ignored by `.gitignore`.
3. If connecting to a cloud database or Google Cloud Platform service account:
   - Place your service account JSON file here as `service-account.json`.
   - Set the corresponding environment variable:
     ```bash
     export GOOGLE_APPLICATION_CREDENTIALS="credentials/service-account.json"
     ```
   - On Windows PowerShell:
     ```powershell
     $env:GOOGLE_APPLICATION_CREDENTIALS = "credentials/service-account.json"
     ```

## Safe Example Template
If configuring a custom service account for GCP/FOIS integration, you may reference the template structure:
```json
{
  "type": "service_account",
  "project_id": "your-railway-project-id",
  "private_key_id": "YOUR_KEY_ID",
  "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n",
  "client_email": "service-account@your-railway-project-id.iam.gserviceaccount.com",
  "client_id": "000000000000000000000",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token"
}
```
Do not place real production keys into this file.
