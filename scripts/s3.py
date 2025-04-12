import logging
import boto3
import argparse
import os
import sys
import threading
from botocore.exceptions import ClientError

boto3.set_stream_logger("", logging.INFO)
session = boto3.session.Session()
s3 = session.client(
    service_name="s3",
    endpoint_url="https://storage.yandexcloud.net",
    region_name="ru-central1",
)


class UploadProgressPercentage(object):
    def __init__(self, filename):
        self._filename = filename
        self._size = float(os.path.getsize(filename))
        self._seen_so_far = 0
        self._lock = threading.Lock()

    def __call__(self, bytes_amount):
        # To simplify, assume this is hooked up to a single filename
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            sys.stdout.write(
                "\r%s  %s / %s  (%.2f%%)"
                % (self._filename, self._seen_so_far, self._size, percentage)
            )
            sys.stdout.flush()


class DownloadProgressPercentage(object):
    def __init__(self, filename, total_size):
        # For displaying the filename, use the basename not the full path
        self._display_filename = os.path.basename(filename)
        self._size = float(total_size)
        self._seen_so_far = 0
        self._lock = threading.Lock()

    def __call__(self, bytes_amount):
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            sys.stdout.write(
                "\r%s  %s / %s  (%.2f%%)"
                % (self._display_filename, self._seen_so_far, self._size, percentage)
            )
            sys.stdout.flush()


def download_folder(bucket, prefix, local_dir):
    os.makedirs(local_dir, exist_ok=True)

    paginator = s3.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)

    download_count = 0

    for page in page_iterator:
        if "Contents" not in page:
            print(f"No objects found with prefix '{prefix}' in bucket '{bucket}'")
            return 0

        for obj in page["Contents"]:
            key = obj["Key"]

            if key.endswith("/"):
                continue

            if prefix.endswith("/"):
                rel_path = key[len(prefix) :]
            else:
                rel_path = key[len(prefix) :] if key.startswith(prefix + "/") else key

            rel_path = rel_path.lstrip("/")

            local_file_path = os.path.join(local_dir, rel_path)

            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

            try:
                file_size = obj["Size"]
                print(f"Downloading {key} to {local_file_path} ({file_size} bytes)")

                s3.download_file(
                    bucket,
                    key,
                    local_file_path,
                    Callback=DownloadProgressPercentage(local_file_path, file_size)
                    if file_size > 0
                    else None,
                )
                print()
                download_count += 1
            except ClientError as e:
                print(f"\nError downloading {key}: {str(e)}")

    return download_count


def main():
    parser = argparse.ArgumentParser(description="Yandex S3 Upload/Download Tool")
    parser.add_argument(
        "--action",
        "-a",
        type=str,
        required=True,
        choices=["upload", "download"],
        help="Action to perform: upload or download",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="For upload: local file to upload. For download: S3 object key or prefix to download.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="For upload: S3 object key. For download: local file path or directory to save.",
    )
    parser.add_argument("--bucket", "-b", type=str, required=True, help="Bucket name.")
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Download recursively from S3 prefix (folder).",
    )

    args = parser.parse_args()

    try:
        if args.action == "upload":
            s3.upload_file(
                args.input,
                args.bucket,
                args.output,
                Callback=UploadProgressPercentage(args.input),
            )
            print(
                f"\nSuccessfully uploaded {args.input} to {args.bucket}/{args.output}"
            )

        elif args.action == "download":
            if args.recursive:
                file_count = download_folder(args.bucket, args.input, args.output)
                print(
                    f"\nSuccessfully downloaded {file_count} files from {args.bucket}/{args.input} to {args.output}"
                )
            else:
                try:
                    response = s3.head_object(Bucket=args.bucket, Key=args.input)
                    file_size = response["ContentLength"]

                    s3.download_file(
                        args.bucket,
                        args.input,
                        args.output,
                        Callback=DownloadProgressPercentage(args.output, file_size),
                    )
                    print(
                        f"\nSuccessfully downloaded {args.bucket}/{args.input} to {args.output}"
                    )
                except ClientError as e:
                    if e.response["Error"]["Code"] == "404":
                        print(
                            f"\nError: The object {args.input} does not exist in bucket {args.bucket}"
                        )
                    else:
                        print(f"\nError: {str(e)}")
                    sys.exit(1)

    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()