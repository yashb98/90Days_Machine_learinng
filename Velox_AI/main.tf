# --- APIS ---
resource "google_project_service" "apis" {
  for_each = toset([
    "compute.googleapis.com", "servicenetworking.googleapis.com",
    "sqladmin.googleapis.com", "run.googleapis.com",
    "redis.googleapis.com", "secretmanager.googleapis.com",
    "aiplatform.googleapis.com"
  ])
  service            = each.key
  disable_on_destroy = false
}

# --- NETWORKING ---
resource "google_compute_network" "vpc_main" {
  name                    = "vpc-main"
  auto_create_subnetworks = false
  depends_on              = [google_project_service.apis]
}

resource "google_compute_subnetwork" "subnet_app" {
  name          = "subnet-app"
  ip_cidr_range = "10.0.0.0/20"
  region        = var.region
  network       = google_compute_network.vpc_main.id
}

resource "google_compute_global_address" "private_ip_alloc" {
  name          = "private-ip-alloc"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 20
  address       = "10.0.16.0"
  network       = google_compute_network.vpc_main.id
}

resource "google_service_networking_connection" "default" {
  network                 = google_compute_network.vpc_main.id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.private_ip_alloc.name]
}

# --- DATABASE (POSTGRES) ---
resource "google_sql_database_instance" "postgres_primary" {
  name             = "ai-voice-db-prod-${random_id.db_suffix.hex}" # Suffix ensures uniqueness
  database_version = "POSTGRES_15"
  region           = var.region
  deletion_protection = false # Set to true for real production

  settings {
    tier    = "db-custom-2-7680" # Lowered for testing; use Enterprise Plus for Prod
    edition = "ENTERPRISE"       # Switch to ENTERPRISE_PLUS for Prod
    
    ip_configuration {
      ipv4_enabled    = false
      private_network = google_compute_network.vpc_main.id
    }
    database_flags {
      name  = "max_connections"
      value = "5000"
    }
  }
  depends_on = [google_service_networking_connection.default]
}

resource "random_id" "db_suffix" {
  byte_length = 4
}

# --- REDIS CACHE ---
resource "google_redis_instance" "redis_cache" {
  name           = "voice-context-cache"
  tier           = "STANDARD_HA"
  memory_size_gb = 5
  region         = var.region
  
  authorized_network = google_compute_network.vpc_main.id
  connect_mode       = "PRIVATE_SERVICE_ACCESS"
}