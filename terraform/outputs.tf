# =============================================================================
# Output Values
# =============================================================================

output "resource_group_name" {
  description = "Name of the Azure resource group"
  value       = azurerm_resource_group.this.name
}

output "databricks_workspace_url" {
  description = "URL of the Databricks workspace"
  value       = "https://${azurerm_databricks_workspace.this.workspace_url}"
}

output "databricks_workspace_id" {
  description = "Azure resource ID of the Databricks workspace"
  value       = azurerm_databricks_workspace.this.id
}

output "databricks_workspace_name" {
  description = "Name of the Databricks workspace"
  value       = azurerm_databricks_workspace.this.name
}

output "metastore_id" {
  description = "Unity Catalog metastore ID"
  value       = databricks_metastore.this.id
}

output "catalog_name" {
  description = "Name of the analytics catalog"
  value       = databricks_catalog.analytics.name
}

output "engineering_cluster_id" {
  description = "ID of the shared engineering cluster"
  value       = databricks_cluster.shared_engineering.id
}

output "engineering_pool_id" {
  description = "ID of the data engineering instance pool"
  value       = databricks_instance_pool.data_engineering.id
}

output "science_pool_id" {
  description = "ID of the data science instance pool"
  value       = databricks_instance_pool.data_science.id
}

output "engineering_policy_id" {
  description = "ID of the data engineering cluster policy"
  value       = databricks_cluster_policy.data_engineering.id
}

output "science_policy_id" {
  description = "ID of the data science cluster policy"
  value       = databricks_cluster_policy.data_science.id
}

output "secret_scope_pipeline" {
  description = "Name of the pipeline secret scope"
  value       = databricks_secret_scope.pipeline.name
}

output "secret_scope_ml_serving" {
  description = "Name of the ML serving secret scope"
  value       = databricks_secret_scope.ml_serving.name
}
