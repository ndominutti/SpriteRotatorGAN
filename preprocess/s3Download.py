import boto3
import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aws-key", type=str)
    parser.add_argument("--aws-pass", type=str)
    parser.add_argument("--aws-region", type=str)
    parser.add_argument("--bucket", type=str)
    parser.add_argument("--folder", type=str)
    parser.add_argument("--local-folder", type=str, default="../data/raw")

    args = parser.parse_args()
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=args.aws_key,
        aws_secret_access_key=args.aws_pass,
        region_name=args.aws_region,
    )

    local_folder = "../data/raw/" + args.folder
    if not os.path.exists(local_folder):
        print(f"Creating local folder: {local_folder}")
        os.makedirs(local_folder, exist_ok=True)

    paginator = s3_client.get_paginator("list_objects_v2")
    print(f"Downloading data into: {local_folder}")
    for page in paginator.paginate(Bucket=args.bucket, Prefix=args.folder):
        for obj in page.get("Contents", []):
            s3_key = obj["Key"]
            if s3_key != args.folder:
                relative_path = os.path.relpath(s3_key, args.folder)
                local_file_path = os.path.join(local_folder, relative_path)
                s3_client.download_file(args.bucket, s3_key, local_file_path)
    print("Success!")
