# =============================================================================
# Input Variables
# =============================================================================

# ---------------------------------------------------------------------------
# General
# ---------------------------------------------------------------------------

variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
  default     = "analytics-platform"
}

variable "environment" {
  description = "Deployment environment (dev, staging, production)"
  type        = string
  default     = "production"

  validation {
    condition     = contains(["dev", "staging", "production"], var.environment)
    error_message = "Environment must be one of: dev, staging, production."
  }
}

# ---------------------------------------------------------------------------
# Azure
# ---------------------------------------------------------------------------

variable "azure_subscription_id" {
  description = "Azure subscription ID"
  type        = string
  sensitive   = true
}

variable "azure_location" {
  description = "Azure region for resource deployment"
  type        = string
  default     = "eastus2"
}

variable "databricks_sku" {
  description = "Databricks workspace SKU (standard or premium)"
  type        = string
  default     = "premium"

  validation {
    condition     = contains(["standard", "premium"], var.databricks_sku)
    error_message = "SKU must be 'standard' or 'premium'."
  }
}

# ---------------------------------------------------------------------------
# Networking
# ---------------------------------------------------------------------------

variable "no_public_ip" {
  description = "Deploy workspace with no public IP (secure cluster connectivity)"
  type        = bool
  default     = true
}

variable "vnet_id" {
  description = "Virtual network ID for VNet-injected workspace"
  type        = string
  default     = ""
}

variable "private_subnet_name" {
  description = "Name of the private subnet for Databricks"
  type        = string
  default     = ""
}

variable "public_subnet_name" {
  description = "Name of the public subnet for Databricks"
  type        = string
  default     = ""
}

variable "private_subnet_nsg_id" {
  description = "NSG association ID for the private subnet"
  type        = string
  default     = ""
}

variable "public_subnet_nsg_id" {
  description = "NSG association ID for the public subnet"
  type        = string
  default     = ""
}

# ---------------------------------------------------------------------------
# Unity Catalog
# ---------------------------------------------------------------------------

variable "unity_catalog_storage_account" {
  description = "Storage account name for Unity Catalog metastore"
  type        = string
}

variable "unity_catalog_container" {
  description = "Storage container name for Unity Catalog metastore"
  type        = string
  default     = "unity-catalog"
}

# ---------------------------------------------------------------------------
# Cluster Configuration
# ---------------------------------------------------------------------------

variable "allowed_node_types" {
  description = "Allowed VM types for data engineering clusters"
  type        = list(string)
  default     = [
    "Standard_DS3_v2",
    "Standard_DS4_v2",
    "Standard_DS5_v2",
    "Standard_E4ds_v5",
    "Standard_E8ds_v5",
  ]
}

variable "ml_node_types" {
  description = "Allowed VM types for data science / ML clusters"
  type        = list(string)
  default     = [
    "Standard_DS4_v2",
    "Standard_DS5_v2",
    "Standard_NC6s_v3",
    "Standard_NC12s_v3",
  ]
}

variable "default_node_type" {
  description = "Default node type for engineering instance pool"
  type        = string
  default     = "Standard_DS4_v2"
}

variable "ml_default_node_type" {
  description = "Default node type for ML instance pool"
  type        = string
  default     = "Standard_DS5_v2"
}

variable "max_workers" {
  description = "Maximum number of workers for engineering clusters"
  type        = number
  default     = 16
}

variable "ml_max_workers" {
  description = "Maximum number of workers for ML clusters"
  type        = number
  default     = 8
}

# ---------------------------------------------------------------------------
# Instance Pools
# ---------------------------------------------------------------------------

variable "pool_min_idle" {
  description = "Minimum idle instances in the engineering pool"
  type        = number
  default     = 1
}

variable "pool_max_capacity" {
  description = "Maximum capacity of the engineering instance pool"
  type        = number
  default     = 20
}

variable "ml_pool_max_capacity" {
  description = "Maximum capacity of the ML instance pool"
  type        = number
  default     = 10
}

# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------

variable "storage_account_key" {
  description = "Azure Storage account key for raw data access"
  type        = string
  sensitive   = true
}

variable "serving_token" {
  description = "Databricks token for model serving endpoints"
  type        = string
  sensitive   = true
}
