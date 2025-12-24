# Boto 3 logic
import boto3
import json
import os
from botocore.exceptions import ClientError, NoCredentialsError
from dotenv import load_dotenv

# Load environment variables (AWS Keys) from the root .env
# We assume the .env is two levels up or in the current execution root
load_dotenv()


def get_s3_client():
    """Helper to initialize the S3 client securely."""
    try:
        return boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
        )
    except Exception as e:
        print(f"❌ Failed to init S3 client: {e}")
        return None


def list_all_buckets() -> str:
    """
    Lists all S3 buckets in the account.
    Returns: JSON string list of bucket names.
    """
    s3 = get_s3_client()
    if not s3:
        return json.dumps({"error": "AWS Credentials Invalid"})

    try:
        response = s3.list_buckets()
        buckets = [b['Name'] for b in response.get('Buckets', [])]
        return json.dumps({"buckets": buckets}, indent=2)
    except ClientError as e:
        return json.dumps({"error": str(e)})


def verify_s3_compliance(bucket_name: str) -> str:
    """
    Deeply inspects an S3 bucket for security compliance (Encryption, Versioning, Public Access).
    """
    s3 = get_s3_client()
    if not s3:
        return json.dumps({"error": "AWS Client Failed"})

    report = {
        "resource": bucket_name,
        "region": s3.meta.region_name,
        "compliance_checks": {
            "encryption": "UNKNOWN",
            "versioning": "UNKNOWN",
            "public_access_block": "UNKNOWN"
        },
        "violations": []
    }

    # 1. Check Encryption
    try:
        enc = s3.get_bucket_encryption(Bucket=bucket_name)
        rules = enc['ServerSideEncryptionConfiguration']['Rules']
        # Look for 'AES256' or 'aws:kms'
        algo = rules[0]['ApplyServerSideEncryptionByDefault']['SSEAlgorithm']
        report['compliance_checks']['encryption'] = algo
    except ClientError:
        # If this call fails, it usually means encryption is NOT enabled
        report['compliance_checks']['encryption'] = "DISABLED"
        report['violations'].append("Server-Side Encryption is DISABLED.")

    # 2. Check Versioning
    try:
        ver = s3.get_bucket_versioning(Bucket=bucket_name)
        # Default is suspended if never enabled
        status = ver.get('Status', 'Suspended')
        report['compliance_checks']['versioning'] = status
        if status != 'Enabled':
            report['violations'].append("Bucket Versioning is NOT Enabled.")
    except ClientError as e:
        report['compliance_checks']['versioning'] = f"Error: {str(e)}"

    # 3. Check Public Access Block (The "Firewall" for S3)
    try:
        pab = s3.get_public_access_block(Bucket=bucket_name)
        conf = pab['PublicAccessBlockConfiguration']
        # We want all of these to be True
        is_secure = all([
            conf['BlockPublicAcls'],
            conf['IgnorePublicAcls'],
            conf['BlockPublicPolicy'],
            conf['RestrictPublicBuckets']
        ])
        report['compliance_checks']['public_access_block'] = "ENABLED" if is_secure else "PARTIAL/DISABLED"
        if not is_secure:
            report['violations'].append(
                "Public Access Blocks are not fully enabled.")
    except ClientError:
        # If no config exists, it defaults to False (Bad!)
        report['compliance_checks']['public_access_block'] = "DISABLED"
        report['violations'].append(
            "No Public Access Block configuration found (Potentially Public!).")

    return json.dumps(report, indent=2)
