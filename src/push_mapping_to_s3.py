from dotenv import load_dotenv
load_dotenv()

import os
import json

DVC_CACHE_DIR = os.path.join(".dvc","cache","files","md5")
ARTIFACT_DVC_PATH = "artifacts.dvc"
S3_BUCKET = os.environ.get("S3_BUCKET_NAME")
print(os.environ)
print("S3_BUCKET:",S3_BUCKET)

def load_dir_hash_from_dvc_file(dvc_file_path):
    import yaml
    with open(dvc_file_path, "r") as f:
        dvc_data = yaml.safe_load(f)
    return dvc_data["outs"][0]["md5"]  # e.g., b7...dir

def load_dir_object(dir_hash):
    if not dir_hash.endswith(".dir"):
        raise ValueError("Hash must end with .dir for directory tracking.")
    prefix = dir_hash[:2]
    suffix = dir_hash[2:]  # remove prefix and ".dir"
    dir_path = os.path.join(DVC_CACHE_DIR, prefix, suffix)
    with open(dir_path, "r") as f:
        return json.load(f)

def build_s3_mapping(dir_listing, s3_bucket):
    mapping = {}
    print('dir_listing:',dir_listing)
    for entry in dir_listing:
        file_name = os.path.basename(entry["relpath"])
        file_hash = entry["md5"]
        s3_key=f"{file_hash[:2]}/{file_hash[2:]}"
        s3_url = f"{s3_bucket}/files/md5/{s3_key}"
        mapping[file_name] = s3_url
    return mapping

# Run the steps
dir_hash = load_dir_hash_from_dvc_file(ARTIFACT_DVC_PATH)
dir_listing = load_dir_object(dir_hash)
mapping = build_s3_mapping(dir_listing, S3_BUCKET)

# Output the result
with open("dvc_artifact_manifest.json", "w") as f:
    json.dump(mapping, f, indent=2)

print(json.dumps(mapping, indent=2))
