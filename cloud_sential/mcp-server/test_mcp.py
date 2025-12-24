# mcp-server/test_mcp.py
from tools.aws_audit import verify_s3_compliance

# The bucket you just created in Step 2
bucket = "acme-data-lake-london-9988"

print(f"🕵️‍♀️ Auditing {bucket} in eu-west-2...")
report = verify_s3_compliance(bucket)
print(report)
