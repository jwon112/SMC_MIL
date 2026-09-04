#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("Usage: Rscript export_core_bounds.R <seurat.rds> <output.csv>")
}

object <- readRDS(args[[1]])
metadata <- as.data.frame(attributes(object)[["meta.data"]], stringsAsFactors = FALSE)
required <- c(
  "Sample", "slide", "patient_id", "x_centroid", "y_centroid",
  "biopsy_cellular_grading", "biopsy_antibody_grading", "biopsy_rejection_type"
)
missing <- setdiff(required, names(metadata))
if (length(missing) > 0) {
  stop(paste("Missing metadata columns:", paste(missing, collapse = ", ")))
}

first_value <- function(values) {
  values <- values[!is.na(values)]
  if (length(values) == 0) "" else as.character(values[[1]])
}

groups <- split(seq_len(nrow(metadata)), metadata$Sample)
rows <- lapply(names(groups), function(sample_id) {
  part <- metadata[groups[[sample_id]], , drop = FALSE]
  data.frame(
    sample_id = sample_id,
    patient_id = first_value(part$patient_id),
    xenium_slide = first_value(part$slide),
    fixed_x_min = min(part$x_centroid, na.rm = TRUE),
    fixed_y_min = min(part$y_centroid, na.rm = TRUE),
    fixed_x_max = max(part$x_centroid, na.rm = TRUE),
    fixed_y_max = max(part$y_centroid, na.rm = TRUE),
    cell_count = nrow(part),
    acr_grade = first_value(part$biopsy_cellular_grading),
    amr_grade = first_value(part$biopsy_antibody_grading),
    rejection_group = first_value(part$biopsy_rejection_type),
    biopsy_timing = if ("biopsy_timing" %in% names(part)) first_value(part$biopsy_timing) else "",
    days_from_transplant = if ("days_from_transplant_to_biopsy" %in% names(part)) first_value(part$days_from_transplant_to_biopsy) else "",
    stringsAsFactors = FALSE
  )
})

result <- do.call(rbind, rows)
dir.create(dirname(args[[2]]), recursive = TRUE, showWarnings = FALSE)
write.csv(result, args[[2]], row.names = FALSE, na = "")
cat(sprintf("[OK] core bounds: %d samples -> %s\n", nrow(result), args[[2]]))
