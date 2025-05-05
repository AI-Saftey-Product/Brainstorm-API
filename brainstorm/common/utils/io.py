from concurrent.futures import ThreadPoolExecutor

from google.cloud import storage
import os
import tarfile
import tempfile


def cloud_storage_upload_string(string, bucket_name, destination_blob_name, client=None):
    if client is None:
        client = storage.Client()

    bucket = client.get_bucket(bucket_name)

    if isinstance(string, list):
        assert len(string) == len(destination_blob_name)
        executor = ThreadPoolExecutor(max_workers=16)

        for _s, _d in zip(string, destination_blob_name):
            executor.submit(cloud_storage_upload_string, string=_s, bucket_name=bucket_name,
                            destination_blob_name=_d, client=client)

        executor.shutdown(wait=True)

    else:
        if destination_blob_name[-4:] == '.png':
            content_type = 'image/png'
        elif destination_blob_name[-5:] == '.json':
            content_type = 'application/json'
        else:
            content_type = 'text/plain'

        bucket.blob(destination_blob_name).upload_from_string(string, content_type=content_type)


def cloud_storage_download_string(bucket_name, destination_blob_name):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    result = bucket.blob(destination_blob_name).download_as_bytes()
    return result





def archive_and_upload(local_dir_path, bucket_name, gcs_prefix):
    """
    Archives a local directory and uploads it to Google Cloud Storage.

    Args:
        local_dir_path (str): Path to the local directory to archive.
        bucket_name (str): Name of the GCS bucket.
        gcs_prefix (str): Prefix/path in the bucket (e.g., 'backups/mydir').
    """
    # Create temporary tar.gz file
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
        archive_path = tmp_file.name

    # Archive directory
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(local_dir_path, arcname=os.path.basename(local_dir_path))

    # Upload to GCS
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"{gcs_prefix}.tar.gz")
    blob.upload_from_filename(archive_path)
    print(f"Uploaded archive to gs://{bucket_name}/{gcs_prefix}.tar.gz")

    # Clean up
    os.remove(archive_path)


def download_and_unpack(bucket_name, gcs_prefix, extract_to_path):
    """
    Downloads an archive from Google Cloud Storage and unpacks it to a directory.

    Args:
        bucket_name (str): Name of the GCS bucket.
        gcs_prefix (str): Prefix/path in the bucket (e.g., 'backups/mydir').
        extract_to_path (str): Local path to extract the archive to.
    """
    # Create temporary tar.gz file
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
        archive_path = tmp_file.name

    # Download from GCS
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"{gcs_prefix}.tar.gz")
    blob.download_to_filename(archive_path)
    print(f"Downloaded archive from gs://{bucket_name}/{gcs_prefix}.tar.gz")

    # Extract the archive
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=extract_to_path)

    # Clean up
    os.remove(archive_path)
