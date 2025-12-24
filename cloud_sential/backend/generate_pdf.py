import os
from reportlab.pdfgen import canvas


def create_dummy_pdf():
    # Get the directory where THIS script is located (backend/)
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Create the data directory path safely
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # Define full file path
    file_path = os.path.join(data_dir, "acme_security_standards_v2.pdf")

    c = canvas.Canvas(file_path)
    c.drawString(100, 800, "ACME CORP CLOUD SECURITY STANDARDS (v2)")

    text = [
        "Section 4.1: S3 Bucket Configuration",
        "All S3 buckets used for 'Production' workloads must adhere to the following strict controls:",
        "1. Encryption: Server-Side Encryption (SSE) must be enabled using AES-256 (SSE-S3).",
        "2. Versioning: Bucket Versioning must be explicitly ENABLED to prevent accidental data loss.",
        "3. Public Access: All 'Block Public Access' settings must be set to TRUE.",
        "",
        "Section 4.2: IAM Users",
        "All IAM users with console access must have Multi-Factor Authentication (MFA) enabled."
    ]

    y = 750
    for line in text:
        c.drawString(100, y, line)
        y -= 20

    c.save()
    print(f"PDF Created successfully at: {file_path}")


if __name__ == "__main__":
    create_dummy_pdf()
