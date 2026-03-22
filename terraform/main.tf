# =============================================================================
# Databricks Analytics Platform — Terraform Infrastructure
# =============================================================================
# Provisions the Databricks workspace, cluster policies, instance pools,
# Unity Catalog metastore, and secret scopes.
#
# Supports both Azure and AWS backends via variable toggle.
# =============================================================================

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    databricks = {
      source  = "databricks/databricks"
      version = "~> 1.40"
    }
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.90"
    }
  }

  backend "azurerm" {
    resource_group_name  = "rg-terraform-state"
    storage_account_name = "stterraformstate"
    container_name       = "tfstate"
    key                  = "databricks-analytics-platform.tfstate"
  }
}

# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

provider "azurerm" {
  features {}
  subscription_id = var.azure_subscription_id
}

provider "databricks" {
  host = azurerm_databricks_workspace.this.workspace_url
}

# ---------------------------------------------------------------------------
# Resource Group
# ---------------------------------------------------------------------------

resource "azurerm_resource_group" "this" {
  name     = "rg-${var.project_name}-${var.environment}"
  location = var.azure_location

  tags = {
    project     = var.project_name
    environment = var.environment
    managed_by  = "terraform"
  }
}

# ---------------------------------------------------------------------------
# Databricks Workspace
# ---------------------------------------------------------------------------

resource "azurerm_databricks_workspace" "this" {
  name                        = "dbw-${var.project_name}-${var.environment}"
  resource_group_name         = azurerm_resource_group.this.name
  location                    = azurerm_resource_group.this.location
  sku                         = var.databricks_sku
  managed_resource_group_name = "rg-${var.project_name}-${var.environment}-managed"

  custom_parameters {
    no_public_ip                                         = var.no_public_ip
    virtual_network_id                                   = var.vnet_id
    private_subnet_name                                  = var.private_subnet_name
    public_subnet_name                                   = var.public_subnet_name
    private_subnet_network_security_group_association_id  = var.private_subnet_nsg_id
    public_subnet_network_security_group_association_id   = var.public_subnet_nsg_id
  }

  tags = {
    project     = var.project_name
    environment = var.environment
    managed_by  = "terraform"
  }
}

# ---------------------------------------------------------------------------
# Unity Catalog Metastore
# ---------------------------------------------------------------------------

resource "databricks_metastore" "this" {
  name          = "metastore-${var.project_name}-${var.environment}"
  storage_root  = "abfss://${var.unity_catalog_container}@${var.unity_catalog_storage_account}.dfs.core.windows.net/"
  region        = var.azure_location
  owner         = "account_admin"

  force_destroy = false

  delta_sharing_scope                       = "INTERNAL"
  delta_sharing_recipient_token_lifetime_in_seconds = 3600
}

resource "databricks_metastore_assignment" "this" {
  metastore_id = databricks_metastore.this.id
  workspace_id = azurerm_databricks_workspace.this.workspace_id
}

resource "databricks_catalog" "analytics" {
  metastore_id = databricks_metastore.this.id
  name         = "analytics_platform"
  comment      = "Analytics platform catalog — medallion architecture"

  properties = {
    environment = var.environment
    team        = "data-engineering"
  }

  depends_on = [databricks_metastore_assignment.this]
}

resource "databricks_schema" "bronze" {
  catalog_name = databricks_catalog.analytics.name
  name         = "bronze"
  comment      = "Raw ingestion layer"

  properties = {
    layer              = "bronze"
    data_classification = "raw"
  }
}

resource "databricks_schema" "silver" {
  catalog_name = databricks_catalog.analytics.name
  name         = "silver"
  comment      = "Cleaned and conformed layer"

  properties = {
    layer              = "silver"
    data_classification = "cleaned"
  }
}

resource "databricks_schema" "gold" {
  catalog_name = databricks_catalog.analytics.name
  name         = "gold"
  comment      = "Business aggregation layer"

  properties = {
    layer              = "gold"
    data_classification = "aggregated"
  }
}

resource "databricks_schema" "ml" {
  catalog_name = databricks_catalog.analytics.name
  name         = "ml"
  comment      = "ML artifacts and predictions"

  properties = {
    layer              = "ml"
    data_classification = "ml_artifacts"
  }
}

# ---------------------------------------------------------------------------
# Cluster Policies
# ---------------------------------------------------------------------------

resource "databricks_cluster_policy" "data_engineering" {
  name = "data-engineering-policy"
  definition = jsonencode({
    "spark_version" : {
      "type" : "allowlist",
      "values" : ["14.3.x-scala2.12", "15.0.x-scala2.12"]
    },
    "node_type_id" : {
      "type" : "allowlist",
      "values" : var.allowed_node_types
    },
    "autoscale.min_workers" : {
      "type" : "range",
      "minValue" : 1,
      "maxValue" : 4
    },
    "autoscale.max_workers" : {
      "type" : "range",
      "minValue" : 2,
      "maxValue" : var.max_workers
    },
    "autotermination_minutes" : {
      "type" : "range",
      "minValue" : 10,
      "maxValue" : 120,
      "defaultValue" : 30
    },
    "custom_tags.team" : {
      "type" : "fixed",
      "value" : "data-engineering"
    },
    "custom_tags.project" : {
      "type" : "fixed",
      "value" : var.project_name
    },
    "spark_conf.spark.databricks.delta.preview.enabled" : {
      "type" : "fixed",
      "value" : "true"
    },
    "spark_conf.spark.databricks.io.cache.enabled" : {
      "type" : "fixed",
      "value" : "true"
    }
  })
}

resource "databricks_cluster_policy" "data_science" {
  name = "data-science-policy"
  definition = jsonencode({
    "spark_version" : {
      "type" : "allowlist",
      "values" : ["14.3.x-cpu-ml-scala2.12", "14.3.x-gpu-ml-scala2.12"]
    },
    "node_type_id" : {
      "type" : "allowlist",
      "values" : var.ml_node_types
    },
    "autoscale.min_workers" : {
      "type" : "range",
      "minValue" : 1,
      "maxValue" : 2
    },
    "autoscale.max_workers" : {
      "type" : "range",
      "minValue" : 2,
      "maxValue" : var.ml_max_workers
    },
    "autotermination_minutes" : {
      "type" : "range",
      "minValue" : 10,
      "maxValue" : 240,
      "defaultValue" : 60
    },
    "custom_tags.team" : {
      "type" : "fixed",
      "value" : "data-science"
    },
    "spark_conf.spark.databricks.delta.preview.enabled" : {
      "type" : "fixed",
      "value" : "true"
    }
  })
}

# ---------------------------------------------------------------------------
# Instance Pools
# ---------------------------------------------------------------------------

resource "databricks_instance_pool" "data_engineering" {
  instance_pool_name = "data-engineering-pool"

  min_idle_instances                  = var.pool_min_idle
  max_capacity                        = var.pool_max_capacity
  node_type_id                        = var.default_node_type
  idle_instance_autotermination_minutes = 15

  preloaded_spark_versions = ["14.3.x-scala2.12"]

  azure_attributes {
    availability       = "ON_DEMAND_AZURE"
    spot_bid_max_price = -1
  }

  custom_tags = {
    team    = "data-engineering"
    project = var.project_name
  }
}

resource "databricks_instance_pool" "data_science" {
  instance_pool_name = "data-science-pool"

  min_idle_instances                  = 0
  max_capacity                        = var.ml_pool_max_capacity
  node_type_id                        = var.ml_default_node_type
  idle_instance_autotermination_minutes = 30

  preloaded_spark_versions = ["14.3.x-cpu-ml-scala2.12"]

  custom_tags = {
    team    = "data-science"
    project = var.project_name
  }
}

# ---------------------------------------------------------------------------
# All-Purpose Clusters
# ---------------------------------------------------------------------------

resource "databricks_cluster" "shared_engineering" {
  cluster_name            = "shared-engineering-${var.environment}"
  spark_version           = "14.3.x-scala2.12"
  instance_pool_id        = databricks_instance_pool.data_engineering.id
  policy_id               = databricks_cluster_policy.data_engineering.id
  data_security_mode      = "USER_ISOLATION"
  single_user_name        = null

  autoscale {
    min_workers = 2
    max_workers = var.max_workers
  }

  autotermination_minutes = 30

  spark_conf = {
    "spark.databricks.delta.preview.enabled"     = "true"
    "spark.databricks.io.cache.enabled"          = "true"
    "spark.databricks.delta.optimizeWrite.enabled" = "true"
    "spark.sql.adaptive.enabled"                  = "true"
  }

  custom_tags = {
    team        = "data-engineering"
    project     = var.project_name
    environment = var.environment
  }
}

# ---------------------------------------------------------------------------
# Secret Scopes
# ---------------------------------------------------------------------------

resource "databricks_secret_scope" "pipeline" {
  name = "pipeline-secrets"
}

resource "databricks_secret_scope" "ml_serving" {
  name = "ml-serving"
}

resource "databricks_secret" "storage_account_key" {
  scope        = databricks_secret_scope.pipeline.name
  key          = "storage-account-key"
  string_value = var.storage_account_key
}

resource "databricks_secret" "serving_token" {
  scope        = databricks_secret_scope.ml_serving.name
  key          = "databricks-token"
  string_value = var.serving_token
}

# ---------------------------------------------------------------------------
# Permissions
# ---------------------------------------------------------------------------

resource "databricks_permissions" "cluster_engineering" {
  cluster_id = databricks_cluster.shared_engineering.id

  access_control {
    group_name       = "data_engineering"
    permission_level = "CAN_RESTART"
  }

  access_control {
    group_name       = "data_science"
    permission_level = "CAN_ATTACH_TO"
  }
}

resource "databricks_permissions" "policy_engineering" {
  cluster_policy_id = databricks_cluster_policy.data_engineering.id

  access_control {
    group_name       = "data_engineering"
    permission_level = "CAN_USE"
  }
}

resource "databricks_permissions" "policy_science" {
  cluster_policy_id = databricks_cluster_policy.data_science.id

  access_control {
    group_name       = "data_science"
    permission_level = "CAN_USE"
  }
}
