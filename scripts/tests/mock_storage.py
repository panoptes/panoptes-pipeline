import os
import shutil
import json


class MockPanStorage(object):

    def __init__(self, project_id='panoptes-survey', bucket_name=None, prefix='', data_dir='scripts/tests/mock_data'):
        self.bucket_name = bucket_name
        self.project_id = project_id
        self.data_dir = data_dir
        self.prefix = prefix

    def list_remote(self, prefix=None):
        """List the names of the remote stored files in the bucket with the given prefix"""
        if not prefix:
            prefix = self.prefix
        files = []
        for (dirpath, dirnames, filenames) in os.walk(self.data_dir):
            if len(filenames) > 0:
                for filename in filenames:
                    fl = dirpath+'/'+filename
                    flname = fl.split(self.data_dir+'/')[1]
                    print("file:",flname)
                    if flname.startswith(prefix):
                        files.append(flname)
        return files

    def upload(self, local_path, remote_path=None):
        """Upload the given file to the local mock storage directory."""
        if remote_path is None:
            remote_path = local_path
        shutil.copyfile(local_path, remote_path)
        return remote_path

    def download(self, remote_path, local_path=None):
        """Download the file from the local mock storage directory."""
        if local_path is None:
            local_path = remote_path
        shutil.copyfile(self.data_dir+'/'+remote_path, local_path)
        return local_path

    def download_string(self, remote_path):
        """Download the file as a string from the local mock storage directory."""
        with open(self.data_dir+'/'+remote_path, 'r') as f:
            data = f.read()
        return data.encode('utf-8')

    def upload_string(self, data, remote_path):
        """Upload the given file as a string to the local mock storage directory."""
        with open(self.data_dir+'/'+remote_path, 'w') as f:
            json.dump(data, f)
        return remote_path
