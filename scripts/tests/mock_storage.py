
class MockPanStorage(object):

    def __init__(self, project_id='panoptes-survey', bucket=None, blobs=None):
        self.bucket = bucket
        self.project_id = project_id
        self.blobs = blobs

    def list_remote(self, prefix=None):
        """List the remote files in the bucket with the given prefix, in the form of blobs."""
        # do i need to mock a blob too, or can this be a list of blobs i pass to the mock?
        remote_files = []
        for blob in self.blobs:
            if blob.name.startswith(prefix):
                remote_files.append(blob)
        return remote_files

    def upload(self, local_path, remote_path=None):
        # do i actually upload the given file anywhere... some local temp storage?
        remote_path = local_path
        return remote_path

    def download(self, remote_path, local_path=None):
        # download file if it exists from local storage or something?
        local_path = remote_path
        return local_path
