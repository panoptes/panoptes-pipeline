import google.auth
from google.cloud import bigquery
from google.cloud import bigquery_storage


def get_bq_clients(project_id='panoptes-exp'):
    """Get Bigquery and BigQueryStorage clients.

    Originally from: https://github.com/googleapis/python-bigquery-storage/
    """

    # Explicitly create a credentials object. This allows you to use the same
    # credentials for both the BigQuery and BigQuery Storage clients, avoiding
    # unnecessary API calls to fetch duplicate authentication tokens.
    credentials, your_project_id = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )

    # Make clients.
    bqclient = bigquery.Client(credentials=credentials, project=project_id, )
    bqstorageclient = bigquery_storage.BigQueryReadClient(credentials=credentials)

    return bqclient, bqstorageclient
