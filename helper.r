# helper.r is a file that contain all common utility and healper function used in our main file model.r
#
manualTypes <- data.frame()


# ************************************************
# NPREPROCESSING_removePunctuation()
#
# INPUT: String - fieldName - name of field
#
# OUTPUT : String - name of field with punctuation removed
# ************************************************
NPREPROCESSING_removePunctuation<-function(fieldName){
  return(gsub("[[:punct:][:blank:]]+", "", fieldName))
}

# ************************************************
# NreadDataset() :
#
# Read a CSV file from working directory
#
# INPUT: string - csvFilename - CSV filename
#
# OUTPUT : data frame - contents of the headed CSV file
# ************************************************
NreadDataset<-function(csvFilename){

  dataset<-read.csv(csvFilename,encoding="UTF-8",stringsAsFactors = FALSE)

  # The field names "confuse" some of the library algorithms
  # As they do not like spaces, punctuation, etc.
  names(dataset)<-NPREPROCESSING_removePunctuation(names(dataset))

  print(paste("CSV dataset",csvFilename,"has been read. Records=",nrow(dataset)))
  return(dataset)
}

# ************************************************
# NPREPROCESSING_initialFieldType() :
#
# Test each field for NUMERIC or SYNBOLIC
#
# INPUT: Data Frame - dataset - data
#
# OUTPUT : Vector - Vector of types {NUMERIC, SYMBOLIC}
# ************************************************
NPREPROCESSING_initialFieldType<-function(dataset){

  field_types<-vector()
  for(field in 1:(ncol(dataset))){

    entry<-which(manualTypes$name==names(dataset)[field])
    if (length(entry)>0){
      field_types[field]<-manualTypes$type[entry]
      next
    }

    if (is.numeric(dataset[,field])) {
      field_types[field]<-TYPE_NUMERIC
    }
    else {
      field_types[field]<-TYPE_SYMBOLIC
    }
  }
  return(field_types)
}


#Apply hot-one encoding
NPREPROCESSING_one_hot_specific <- function(dataset, column_name) {
  # Convert the specified column into a factor
  factor_column <- factor(dataset[[column_name]])

  # Generate one-hot encoded columns using model.matrix
  one_hot_encoded <- data.frame(model.matrix(~ factor_column - 1, data = dataset))

  # Rename the one-hot encoded columns
  colnames(one_hot_encoded) <- gsub("factor_column", column_name, colnames(one_hot_encoded))

  # Combine the one-hot encoded columns with the original dataset
  dataset <- cbind(dataset, one_hot_encoded)

  # Optionally, remove the original column to avoid multicollinearity
  dataset[[column_name]] <- NULL

  return(dataset)
}


encode_and_scale_column <- function(column, ordered_categories) {
  # Encoding the categorical variable
  encoded_values <- as.integer(factor(column, levels = ordered_categories)) - 1

  return(encoded_values)
}

# plot histograms for the numeric values...
# different for continuous and discrete
plot_histograms <- function(numeric_dataset) {
  for (field in names(numeric_dataset)) {
    # Create a histogram with 10 bins
    hist(numeric_dataset[[field]], breaks = 10, plot = TRUE,
         main = paste("Histogram of", field),
         xlab = field, ylab = "Frequency",
         col = "lightblue", border = "black", xaxs = "i", yaxs = "i")
  }
}

# show box plots for categorical variables
plot_categorical_barplots <- function(symbolic_dataset, symbolic_fields) {
  for (field in symbolic_fields) {
    num_levels <- length(levels(factor(symbolic_dataset[[field]])))
    # Create bar plot
    barplot(table(symbolic_dataset[[field]]),
            main = paste("Bar Plot of", field),
            xlab = field, ylab = "Count",
            col = rainbow(num_levels))
  }
}

#correlation matrix for numeric values
NPLOT_correlagram<-function(cr){
  short_names <- substring(rownames(cr), 1, 20)
  rownames(cr) <- short_names
  colnames(cr) <- short_names

  #Defines the colour range
  col<-colorRampPalette(c("green", "red"))
  corrplot::corrplot(abs(cr),method="square",
                     order="FPC",
                     cl.ratio=0.2,
                     cl.align="r",
                     tl.cex = 0.6,cl.cex = 0.6,
                     mar=c(1,1,1,1),bty="n")
}

#
count_rows_with_unknowns <- function(data) {
  row_unknown_counts <- apply(data, 1, function(row) {
    sum(row == "Unknown") > 0
  })
  sum(row_unknown_counts)
}

count_unknowns <- function(data) {
  unknown_counts <- sapply(data, function(column) {
    sum(column == "Unknown")
  })
  return(unknown_counts)
}

# use chi test to get rid of unknowns
NPREPROCESSING_unknown_analysis <- function(categorical_data) {
  results <- list()

  for (field in 1:ncol(categorical_data)) {
    column_data <- categorical_data[, field]

    # Create a contingency table comparing 'Unknown' vs. other values
    is_unknown <- column_data == "Unknown"
    table_data <- table(is_unknown, column_data)

    # Perform the Chi-Squared Test
    chi_test_result <- chisq.test(table_data)

    # Store the results
    results[[colnames(categorical_data)[field]]] <- chi_test_result
  }

  return(results)
}

#use interquartile range to identify outliers and then replace them using winsorized
calculate_iqr_outliers <- function(dataset, numeric_fields) {
  winsorized_dataset <- dataset
  for (field in numeric_fields) {
    # Calculate IQR
    iqr_value <- IQR(dataset[[field]])
    # Calculate the first and third quartiles
    q1 <- quantile(dataset[[field]], 0.25)
    q3 <- quantile(dataset[[field]], 0.75)
    # Calculate upper and lower thresholds for outliers
    upper_threshold <- q3 + 1.5 * iqr_value
    lower_threshold <- q1 - 1.5 * iqr_value
    # Print the thresholds
    cat("Field:", field)
    cat("  Upper Threshold for Outliers:", upper_threshold, "\n")
    cat("  Lower Threshold for Outliers:", lower_threshold, "\n")

    # identify outlier values
    upper_outliers <- sum(dataset[[field]] > upper_threshold)
    lower_outliers <- sum(dataset[[field]] < lower_threshold)
    outliers <- upper_outliers + lower_outliers
    #print field name and number of outliers to be replaced
    cat("Number of outliers to be replaced:", outliers, "\n\n")

    # Calculate proportions for winsorization
    upper_probs <- 1-(upper_outliers / length(dataset[[field]]))
    lower_probs <- lower_outliers / length(dataset[[field]])

    # Apply Winsorization if outliers exist
    if (outliers> 0) {
      winsorized_dataset[[field]] <- Winsorize(dataset[[field]], probs = c(lower_probs, upper_probs))

    }
    return(winsorized_dataset)
  }}



#scatterplot for
scatterplot_cor<-function(cr_matrix){
  # Define a threshold for high correlation
  threshold <- 0.3

  # Find indices of high correlations in the correlation matrix
  cor_index <- which(abs(cr_matrix) > threshold, arr.ind = TRUE)

  # Filter out the diagonal and lower triangle to avoid redundancy
  cor_index <- cor_index[cor_index[, 1] < cor_index[, 2], ]

  # Loop over each pair of highly correlated variables and create a scatter plot
  for (index in 1:nrow(cor_index)) {
    # Extract the variable names based on the matrix indices
    var1 <- rownames(cr_matrix)[cor_index[index, 1]]
    var2 <- colnames(cr_matrix)[cor_index[index, 2]]

    # Create the scatter plot
    plot(numeric_dataset[[var1]], numeric_dataset[[var2]],
         main = paste("Scatter plot of", var1, "vs", var2),
         xlab = var1,
         ylab = var2,
         pch = 19,  # Type of point
         col = rgb(0, 0, 1, alpha = 0.5)) # Blue semi-transparent points
  }
}

# barplot comparison for percentage value
percentage_comparison_barplots <- function(dataset, symbolic_fields) {
  # Mark rows as 'With Unknown' or 'Without Unknown'
  dataset$RowType <- apply(dataset, 1, function(row) {
    if (any(row == "Unknown")) "With Unknown" else "Without Unknown"
  })

  for (field in symbolic_fields) {
    # Create a contingency table, exclude 'Unknown' values
    dataset_filtered <- subset(dataset, dataset[[field]] != "Unknown")
    contingency_table <- table(dataset_filtered[[field]], dataset_filtered$RowType)

    # Convert counts to percentages
    contingency_table_pct <- prop.table(contingency_table, margin = 2) * 100

    # Plot
    barplot(contingency_table_pct, beside = TRUE,
            main = paste("Comparison of", field, "(%)"),
            xlab = "Categories", ylab = "%",
            col = rainbow(nrow(contingency_table_pct)),
            legend = rownames(contingency_table_pct),
            names.arg = colnames(contingency_table_pct))
  }
}

# bar plot comparison for unknown
compare_histograms_with_unknown <- function(dataset, numeric_fields) {
  # Mark rows as 'With Unknown' or 'Without Unknown'
  dataset$RowType <- apply(dataset, 1, function(row) {
    if (any(row == "Unknown")) "With Unknown" else "Without Unknown"

  })

  # Create histograms for each numeric field
  for (field in numeric_fields) {
      # Create a layout for side-by-side histograms
      par(mfrow = c(1, 2))

      # Histogram for 'Without Unknown'
      hist(subset(dataset, RowType == "Without Unknown")[[field]],
           breaks = 10, plot = TRUE, main = paste("Without Unknown"),
           xlab = field, ylab = "Frequency", col = "lightblue", border = "black")
     
      # Histogram for 'With Unknown'
      hist(subset(dataset, RowType == "With Unknown")[[field]],
           breaks = 10, plot = TRUE, main = paste("With Unknown"),
           xlab = field, ylab = "Frequency", col = "lightcoral", border = "black")
      
      # Reset layout
      par(mfrow = c(1, 1))
  }
}


# common function to display confusion matrix
show_confusion_matrix<- function(conf_matrix, title) {

  # Convert the confusion matrix to a data frame
  conf_matrix_df <- as.data.frame(conf_matrix$table)
  conf_matrix_df$Label <- c("True Negative", "False Positive", "False Negative", "True Positive")

  # Plot the confusion matrix
  conf_matrix_plot <- ggplot(data = conf_matrix_df, aes(x = Prediction, y = Reference, fill = Freq)) +
    geom_tile(color = "black") +  # Add border
    geom_text(aes(label = paste(Label, "\n", Freq)), size = 6) +  # Increase text size and add labels
    scale_fill_gradient(low = "white", high = "blue") +
    theme_minimal() +
    labs(title = title, x = "Predicted", y = "Actual")

  # Display the plot
  print(conf_matrix_plot)
}

# Plot ROC-AUC Curve for the models
plot_roc_curve <- function(predictions, y_true, title) {
  # Convert factor predictions to numeric if needed
  if (is.factor(predictions)) {
    predictions <- as.numeric(predictions)
  }
  # Compute ROC curve
  roc_data <- roc(y_true, predictions)
  
  # Plot ROC-AUC curve
  plot(roc_data, main = title, col = "blue", lwd = 2,
       cex.main = 0.8, cex.lab = 0.8, cex.axis = 0.8)
  abline(a = 0, b = 1, lty = 2, col = "gray")
  
  # Add AUC value to the plot
  text(0.8, 0.2, paste("AUC =", round(auc(roc_data), 2)),
       col = "blue", cex = 0.8)
}

# create bar_chart for model comparison
plot_bar_compare <- function(com_data, bar_title){
  # bar chart for comparison
  ggplot(com_data, aes(x = Model, y = Value, fill = Metric)) +
    geom_bar(stat = "identity", position = "dodge") +
    labs(title = bar_title,
         y = "Metric Value",
         x = "ML Algorithms") +
    theme_minimal()
}

