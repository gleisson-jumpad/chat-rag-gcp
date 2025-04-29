# PostgreSQL Connection Tester for Cloud Run

This application demonstrates connecting to a PostgreSQL instance on Google Cloud SQL from Cloud Run using various connection methods.

## Setup and Deployment

### Local Testing

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download the Cloud SQL Auth Proxy:**
   For macOS:
   ```bash
   curl -o cloud-sql-proxy https://storage.googleapis.com/cloud-sql-connectors/cloud-sql-proxy/v2.15.2/cloud-sql-proxy.darwin.arm64
   chmod +x cloud-sql-proxy
   ```
   For other platforms, see: [Cloud SQL Auth Proxy Installation](https://cloud.google.com/sql/docs/postgres/connect-auth-proxy#install)

3. **Set environment variables:**
   ```bash
   export INSTANCE_CONNECTION_NAME="your-project:region:instance-name"
   export PG_DB="your-database-name"
   export PG_USER="your-database-user"
   export PG_PASSWORD="your-database-password"
   export DB_PUBLIC_IP="your-postgresql-public-ip"  # Only if using direct connection
   ```

4. **Run the application:**
   ```bash
   streamlit run app/main.py
   ```

### Cloud Run Deployment

1. **Update your project information** in `cloud-run-deploy.sh`:
   - Set your `PROJECT_ID`
   - Set your desired `REGION`
   - Verify the `INSTANCE_CONNECTION_NAME`
   - Update database credentials
   
2. **Make the deployment script executable:**
   ```bash
   chmod +x cloud-run-deploy.sh
   ```

3. **Run the deployment script:**
   ```bash
   ./cloud-run-deploy.sh
   ```

4. **Alternative: Deploy manually with gcloud**
   ```bash
   # Build the container
   gcloud builds submit --tag gcr.io/PROJECT_ID/postgres-connection-test
   
   # Deploy to Cloud Run
   gcloud run deploy postgres-connection-test \
     --image gcr.io/PROJECT_ID/postgres-connection-test \
     --platform managed \
     --region REGION \
     --set-env-vars="INSTANCE_CONNECTION_NAME=PROJECT:REGION:INSTANCE" \
     --set-env-vars="PG_DB=postgres" \
     --set-env-vars="PG_USER=postgres" \
     --set-env-vars="PG_PASSWORD=your_password_here" \
     --allow-unauthenticated \
     --add-cloudsql-instances=PROJECT:REGION:INSTANCE
   ```

## Important Notes for Cloud Run

1. **Cloud SQL Auth Proxy:** In Cloud Run, the Cloud SQL Auth Proxy is automatically set up when you use the `--add-cloudsql-instances` flag.

2. **Unix Socket Path:** In Cloud Run, the socket is available at `/cloudsql/INSTANCE_CONNECTION_NAME`.

3. **PostgreSQL Connection:** When connecting to PostgreSQL through the unix socket, use:
   ```python
   conn = psycopg2.connect(
       dbname=os.getenv("PG_DB"),
       user=os.getenv("PG_USER"),
       password=os.getenv("PG_PASSWORD"),
       host="/cloudsql/INSTANCE_CONNECTION_NAME"
   )
   ```

4. **Security:** Make sure your database credentials are passed securely as environment variables. Consider using Secret Manager for production deployments.

## Troubleshooting

If you encounter connection issues:

1. Verify your Cloud SQL instance has the Cloud Run service account in its authorized principals.
2. Make sure the service account has the necessary IAM permissions.
3. Check if the Cloud SQL instance is in the same region as your Cloud Run service.
4. Verify that your database user and password are correct.
5. Look for the connection error message in the app or in the Cloud Run logs. 