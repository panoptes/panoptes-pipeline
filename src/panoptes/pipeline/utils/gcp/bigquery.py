import google.auth
from google.cloud import bigquery
from google.cloud import bigquery_storage
import pandas_gbq


def get_bq_clients():
    """Get Bigquery and BigQueryStorage clients.

    Originally from: https://github.com/googleapis/python-bigquery-storage/
    """

    # Explicitly create a credentials object. This allows you to use the same
    # credentials for both the BigQuery and BigQuery Storage clients, avoiding
    # unnecessary API calls to fetch duplicate authentication tokens.
    credentials, project_id = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )

    # Make clients.
    bqclient = bigquery.Client(credentials=credentials, project=project_id, )
    bqstorageclient = bigquery_storage.BigQueryReadClient(credentials=credentials)

    return bqclient, bqstorageclient


def to_bigquery(df, table_id='images.sources', project_id='project-panoptes-01'):
    """Save a dataframe to a BigQuery table."""
    pandas_gbq.to_gbq(df, table_id, project_id=project_id, if_exists='append')
