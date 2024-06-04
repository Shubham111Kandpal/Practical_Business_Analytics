rm(list=ls())
cat("\014")

#Load libraries
library(pacman)
library(glmnet)
library(nnet)
library(pROC)
library(knitr)
MYLIBRARIES<-c("outliers", "corrplot","MASS","formattable","stats",
               "PerformanceAnalytics","scatterplot3d","caret", "psych",
               "dplyr", "smotefamily", "xgboost", "e1071",
               "rpart", "rpart.plot", "randomForest", "ggplot2", "knitr", "kableExtra", "DescTools", "VIM", "ANN", "GA", "neuralnet")

pacman::p_load(char= MYLIBRARIES,install=TRUE,character.only=TRUE)

DATASET_FILENAME  <- "BankChurners.csv"   # Name of input dataset file
MAX_LITERALS      <- 55                   # Maximum number of 1-hot encoding new fields
TYPE_NUMERIC      <- "NUMERIC"            # field is initially a numeric
TYPE_SYMBOLIC     <- "SYMBOLIC"           # field is a string
DISCRETE_BINS     <- 6                    # Number of empty bins to determine discrete
TYPE_DISCRETE     <- "DISCRETE"           # field is discrete (numeric)
TYPE_ORDINAL      <- "ORDINAL"            # field is continuous numeric

set.seed(1234)

# loading all utility functions
source("helper.r")

######################################Data Analysis & Pre-processing - [START]#################################
# Read and preprocess the dataset
dataset <- NreadDataset("BankChurners.csv")
head(dataset)

# check if there are duplicates in CLIENTNUM
duplicates <- dataset[duplicated(dataset$CLIENTNUM), ]
print(paste("Number of duplicate rows: ", nrow(duplicates)))
dataset <- dataset[!duplicated(dataset$CLIENTNUM), ]

# Dropping the following columns because we are not using them
# CLIENTNUM, NaiveBayesClassifierAttritionFlagCardCategoryContactsCount12monDependentcountEducationLevelMonthsInactive12mon1,
# and NaiveBayesClassifierAttritionFlagCardCategoryContactsCount12monDependentcountEducationLevelMonthsInactive12mon2
columns_to_remove <- c(
  "CLIENTNUM",
  "NaiveBayesClassifierAttritionFlagCardCategoryContactsCount12monDependentcountEducationLevelMonthsInactive12mon1",
  "NaiveBayesClassifierAttritionFlagCardCategoryContactsCount12monDependentcountEducationLevelMonthsInactive12mon2"
)
dataset <- dataset[ , !(names(dataset) %in% columns_to_remove)]

# Checking dataset columns/fields for NUMERIC or SYNBOLIC
field_types <- NPREPROCESSING_initialFieldType(dataset)
numeric_fields<-names(dataset)[field_types=="NUMERIC"]
print(paste("NUMERIC FIELDS=",length(numeric_fields)))
print(numeric_fields)

symbolic_fields<-names(dataset)[field_types=="SYMBOLIC"]
print(paste("SYMBOLIC FIELDS=",length(symbolic_fields)))
print(symbolic_fields)

# Splitting the dataset into numeric and symbolic datasets
numeric_dataset <- dataset[, field_types == "NUMERIC"]
symbolic_dataset <- dataset[, field_types == "SYMBOLIC"]
head(numeric_dataset)
head(symbolic_dataset)

# plot histograms for the numeric values
plot_histograms(numeric_dataset)

# plot box plots for categorical variables
plot_categorical_barplots(symbolic_dataset, symbolic_fields)

# plot correlation matrix for numeric variables
cr_matrix <- cor(numeric_dataset, use = "complete.obs")
print(cr_matrix)
NPLOT_correlagram(cr_matrix)
scatterplot_cor(cr_matrix)

# make bar plots for comparison of symbolic fields
percentage_comparison_barplots(dataset, symbolic_fields)

print(count_unknown<-count_unknowns(dataset))
print(count_rows_with_unknown<-count_rows_with_unknowns(dataset))
print(unknown_analysis<-NPREPROCESSING_unknown_analysis(symbolic_dataset))

# make bar plots for comparison of data with unknown
compare_histograms_with_unknown(dataset, numeric_fields)

# remove the  rows with unknown values
dataset <- dataset[!apply(dataset, 1, function(row) any(row == "Unknown")), ]
head(dataset)

for (field in numeric_fields) {
  # Create the plot for each field
  p <- ggplot(dataset, aes_string(x = field, y = field)) +
    geom_boxplot(fill = "lightgreen", width = 0.2) +
    theme_classic() +
    labs(title = paste("Box Plot with Dot Plot of", field), x = field, y = "Values")

  # Print the plot
  print(p)
}

# use interquartile range to identify outliers and then replace them using winsorized
dataset <- calculate_iqr_outliers(dataset, numeric_fields)

# change MaritalStatus column into one hot_encoding assign new values
dataset <- NPREPROCESSING_one_hot_specific(dataset, "MaritalStatus")

# Hierarchical conversion from character to numeric for other symbolic fields.
EducationLevel <- c("Uneducated", "High School", "College", "Graduate", "Post-Graduate", "Doctorate")
IncomeCategory <- c("Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "$120K +")
CardCategory <- c("Blue", "Silver", "Gold", "Platinum")
Gender <- c("F", "M")
AttritionFlag <- c("Attrited Customer", "Existing Customer")

# Applying encoding to categorical columns
dataset$EducationLevel <- encode_and_scale_column(dataset$EducationLevel, EducationLevel)
dataset$IncomeCategory <- encode_and_scale_column(dataset$IncomeCategory, IncomeCategory)
dataset$CardCategory <- encode_and_scale_column(dataset$CardCategory, CardCategory)
dataset$Gender <- encode_and_scale_column(dataset$Gender, Gender)
dataset$AttritionFlag <- encode_and_scale_column(dataset$AttritionFlag, AttritionFlag)

columns_to_remove <- c(
  "MaritalStatusSingle"
)
dataset <- dataset[ , !(names(dataset) %in% columns_to_remove)]

# Normalization using log
dataset$CreditLimit <- log10(dataset$CreditLimit + 1)
dataset$AvgOpenToBuy <- log10(dataset$AvgOpenToBuy + 1)
dataset$TotalTransAmt <- log10(dataset$TotalTransAmt + 1)
dataset$TotalAmtChngQ4Q1 <- log10(dataset$TotalAmtChngQ4Q1 + 1)

# Normalization using square root
dataset$TotalCtChngQ4Q1 <- sqrt(dataset$TotalCtChngQ4Q1)
dataset$TotalRevolvingBal <- sqrt(dataset$TotalRevolvingBal)

# Normalization using Arcsine square root
dataset$AvgUtilizationRatio <- asin(sqrt(dataset$AvgUtilizationRatio))

# Normalization using Z-scale
fields_to_normalise <- c("CreditLimit", "TotalRevolvingBal", "AvgOpenToBuy", "TotalAmtChngQ4Q1", "TotalTransAmt", "TotalCtChngQ4Q1", "AvgUtilizationRatio")
dataset[fields_to_normalise] <- as.data.frame(scale(dataset[fields_to_normalise], center = TRUE, scale = TRUE))

cr_matrix2 <- cor(dataset, use = "complete.obs")
NPLOT_correlagram(cr_matrix2)
print(cr_matrix2)

# Summary Statistics
print(summary(dataset))

# Shuffling the dataset
shuffled_dataset <- dataset[sample(nrow(dataset)), ]

######################################Data Analysis & Preprocessing - [END]######################

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
######################################[Splitting Dataset]########################################

# Splitting the dataset into training and testing sets (70% training, 30% testing)
train_size <- round(0.7 * nrow(shuffled_dataset))
training_data <- shuffled_dataset[1:train_size, ]
testing_data <- shuffled_dataset[(train_size + 1):nrow(shuffled_dataset), ]

# Balancing the training data
training_data$AttritionFlag <- as.factor(training_data$AttritionFlag)
training_data_balanced <- SMOTE(X = training_data[, -which(names(training_data) == "AttritionFlag")], target = training_data$AttritionFlag,
                                K = 10,
                                dup_size = 0)

balanced_dataset <- training_data_balanced$data
names(balanced_dataset)[names(balanced_dataset) == "class"] <- "AttritionFlag"
balanced_dataset$AttritionFlag <- as.factor(balanced_dataset$AttritionFlag)
print(symbolic_fields)

# Processing type correctly
fields_to_round <- c("TotalRelationshipCount","AttritionFlag","CardCategory", "Dependentcount", "TotalTransCt", "CustomerAge", "Monthsonbook", "Gender", "EducationLevel", "ContactsCount12mon", "IncomeCategory", "MonthsInactive12mon", "MaritalStatusDivorced", "MaritalStatusMarried")
balanced_dataset <- data.frame(sapply(balanced_dataset, function(x) as.numeric(as.character(x))))
str(balanced_dataset)

balanced_dataset[fields_to_round] <- lapply(balanced_dataset[fields_to_round], round)
print(names(balanced_dataset))
str(balanced_dataset)
balanced_dataset <- balanced_dataset[sample(nrow(balanced_dataset)), ]

# Split the data into features and target for machine learning models
features <- balanced_dataset[, !names(balanced_dataset) %in% "AttritionFlag"]
target <- as.factor(balanced_dataset$AttritionFlag)

# Convert data frames to matrices
X_train <- as.matrix(balanced_dataset[, !names(balanced_dataset) %in% "AttritionFlag"])
y_train <- as.numeric(balanced_dataset$AttritionFlag)

X_test <- as.matrix(testing_data[, !names(testing_data) %in% "AttritionFlag"])
y_test <- as.numeric(testing_data$AttritionFlag)
#######################################################################################

#################################  Feature selection  ##################################

#split data into features and target

test_features <- testing_data[, !names(testing_data) %in% "AttritionFlag"]
test_target <- as.factor(testing_data$AttritionFlag)

# Fitting a logistic regression model
logit_model <- glm(AttritionFlag ~ ., data = training_data, family = "binomial")

# Making predictions
pred <- predict(logit_model, newdata = test_features, type = "response")

#use the logistic regression to work out odds_ratios_df preparation code
odds_ratios <- exp(coef(logit_model))
odds_ratios_df <- as.data.frame(odds_ratios)
odds_ratios_df$Feature <- rownames(odds_ratios_df)
colnames(odds_ratios_df)[1] <- "OddsRatio"
odds_ratios_df <- odds_ratios_df[order(-odds_ratios_df$OddsRatio), ]

# Display the ranked odds ratios
print(odds_ratios_df)

# Plotting Odds Ratios with ordered Features
ggplot(odds_ratios_df, aes(x = reorder(Feature, OddsRatio), y = OddsRatio)) +
  geom_col() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Logistic Regression Odds Ratios", x = "Feature", y = "Odds Ratio")

cr_matrix3 <- cor(dataset, use = "complete.obs")
NPLOT_correlagram(cr_matrix3)
print(cr_matrix3)

#2 approach feature selection
# Calculate the Pearson correlation
correlation_matrix <- cor(shuffled_dataset, method = "pearson")

# Extract the correlations related to 'Attrition_Flag'
attrition_correlations <- correlation_matrix["AttritionFlag",]

# Sort and display the correlations
sorted_correlations <- sort(attrition_correlations, decreasing = TRUE)


# Plot the correlation matrix
corrplot(correlation_matrix, method = "circle", type = "full",
         order = "hclust", tl.col = "blue", tl.srt = 45)

# Print the correlations
print(sorted_correlations)


############################# ML Model implementation #################################
#######################################################################################
#################################SVM - [FIN]###########################################

# Train a simple SVM model using linear kernel
svm_model <- svm(features, target, kernel = "linear")

# Make predictions on the test set
test_predictions <- predict(svm_model, test_features)

# Evaluate performance
conf_matrix <- confusionMatrix(factor(test_predictions), test_target)
print(conf_matrix)

# Extracting various performance metrics
accuracy <- conf_matrix$overall["Accuracy"]
precision <- conf_matrix$byClass["Precision"]
recall <- conf_matrix$byClass["Sensitivity"]
f1_score <- conf_matrix$byClass["F1"]

# Evaluate performance
conf_matrix <- confusionMatrix(factor(test_predictions), factor(test_target))
print(conf_matrix)

# Extract various performance metrics
accuracy_spl_svm <- conf_matrix$overall["Accuracy"]
precision_spl_svm <- conf_matrix$byClass["Precision"]
recall_spl_svm <- conf_matrix$byClass["Sensitivity"]
f1_score_spl_svm <- conf_matrix$byClass["F1"]

# Print the metrics
cat("Accuracy:", accuracy_spl_svm, "\n")
cat("Precision:", precision_spl_svm, "\n")
cat("Recall (Sensitivity):", recall_spl_svm, "\n")
cat("F1 Score:", f1_score_spl_svm, "\n")

# Create a title string including the accuracy
title <- sprintf("Confusion Matrix - simple SVM Model - Accuracy: %.2f%%", accuracy_spl_svm * 100)
show_confusion_matrix(conf_matrix, title = title)

####################################################################################
#Train an svmradial SVM with a grid search and train control

training_data_svm<-balanced_dataset
testing_data_svm<-testing_data

training_data_svm$AttritionFlag <- factor(training_data_svm$AttritionFlag, levels = c(0, 1), labels = c("X0", "X1"))
testing_data_svm$AttritionFlag <- factor(testing_data_svm$AttritionFlag, levels = c(0, 1), labels = c("X0", "X1"))

print(training_data_svm)
#class_weights <- ifelse(training_data_svm$AttritionFlag == 0, 80, 20)

# Define training control for cross-validation
train_control <- trainControl(method = "cv", number = 5, summaryFunction = prSummary, classProbs = T, verboseIter = T)

# Define the grid of hyperparameters for SVM
hyper_grid <- expand.grid(C = c(0.1, 1, 10), sigma = c(0.01, 0.1, 1))

# Train the model using grid search and cross-validation
svm_model <- train(x = training_data_svm[, !names(training_data_svm) %in% "AttritionFlag"], y = training_data_svm$AttritionFlag,
                   method = "svmRadial", metric = "F", trControl = train_control, tuneGrid = hyper_grid)

# Examine the best hyperparameters
print(svm_model$bestTune)

# Prepare test set for evaluation
test_features_svm <- testing_data_svm[, !names(testing_data_svm) %in% "AttritionFlag"]
test_target_svm <-  as.factor(testing_data_svm$AttritionFlag)

# Evaluate the final model on the test set
test_predictions <- predict(svm_model, test_features_svm)

# Evaluate performance
conf_matrix <- confusionMatrix(test_predictions, test_target_svm)
print(conf_matrix)

# Extract various performance metrics
accuracy_svm <- conf_matrix$overall["Accuracy"]
precision_svm <- conf_matrix$byClass["Precision"]
recall_svm <- conf_matrix$byClass["Sensitivity"]
f1_score_svm <- conf_matrix$byClass["F1"]

# Print the metrics
cat("Accuracy:", accuracy_svm, "\n")
cat("Precision:", precision_svm, "\n")
cat("Recall (Sensitivity):", recall_svm, "\n")
cat("F1 Score:", f1_score_svm, "\n")

# Create a title string including the accuracy
title <- sprintf("Confusion Matrix - SVM Model - Accuracy: %.2f%%", accuracy_svm * 100)
show_confusion_matrix(conf_matrix, title = title)


###########################################################################################
#####################[[Neural Network with GA]- [FIN]] ##################
balanced_dataset_saved <- balanced_dataset
features_train <- as.matrix(balanced_dataset[, !names(balanced_dataset) %in% "AttritionFlag"])
target_train <- as.numeric(balanced_dataset$AttritionFlag)

features_test <- as.matrix(testing_data[, !names(testing_data) %in% "AttritionFlag"])
target_test <- as.numeric(testing_data$AttritionFlag)

target_train <- factor(target_train)
target_test <- factor(target_test)

fitness_nn <- function(params) {
  size = round(params[1])  # Number of neurons in the layer
  decay = params[2]        # Decay rate
  set.seed(123)

  nn_model <- train(
    x = features_train, y = target_train,
    method = "nnet",
    preProcess = c("center", "scale"),
    tuneGrid = expand.grid(size = size, decay = decay),
    trace = FALSE
  )

  best_acc <- max(nn_model$results$Accuracy)
  return(best_acc)
}

# Adjusted upper and lower bounds for the size and decay
lbound <- c(1, 0.0001)
ubound <- c(3, 1)

# Genetic Algorithm setup
GA_result <- ga(type = "real-valued", fitness = fitness_nn,  lower = lbound, upper = ubound, popSize = 10, maxiter = 5, run = 5)

# Extract the best parameters
best_params <- GA_result@solution

best_size <- round(best_params[1])
best_decay <- best_params[2]

best_nn_model <- train(
  x = features_train, y = target_train,
  method = "nnet",
  preProcess = c("center", "scale"),
  tuneGrid = expand.grid(size = best_size, decay = best_decay),
  trace = FALSE)

# Print best parameters
print(best_params)

predictions_nn_FIN <- predict(best_nn_model, features_test)

# Evaluate performance
conf_matrix_nn_FIN <- confusionMatrix(predictions_nn_FIN, target_test)
print(conf_matrix_nn_FIN)

# Extracting various performance metrics
accuracy_nn_FIN <- conf_matrix_nn_FIN$overall["Accuracy"]
precision_nn_FIN <- conf_matrix_nn_FIN$byClass["Precision"]
recall_nn_FIN <- conf_matrix_nn_FIN$byClass["Sensitivity"]
f1_score_nn_FIN <- conf_matrix_nn_FIN$byClass["F1"]

# Print the metrics for the neural network model
cat("Accuracy:", accuracy_nn_FIN, "\n")
cat("Precision:", precision_nn_FIN, "\n")
cat("Recall (Sensitivity):", recall_nn_FIN, "\n")
cat("F1 Score:", f1_score_nn_FIN, "\n")

# Display confusion Matrix for FIN's Neural Network
title_nn_FIN <- sprintf("Confusion Matrix - FIN Neural Network - Accuracy: %.2f%%", accuracy_nn_FIN * 100)
show_confusion_matrix(conf_matrix_nn_FIN, title = title_nn_FIN)

title_nn_roc <- sprintf("ROC Curve - FIN Neural Network - Accuracy: %.2f%%", accuracy_nn_FIN * 100)
# Plot ROC-AUC Curve
plot_roc_curve(predictions_nn_FIN, y_test, title=title_nn_roc)

balanced_dataset<-balanced_dataset_saved
test_features <- testing_data[, !names(testing_data) %in% "AttritionFlag"]
test_target <- as.factor(testing_data$AttritionFlag)

##########################################[END]##################################################
########################## Decision Tree Model Training and Evaluation - [SHEILA] ###############
#################################################################################################
# Build the Decision Tree model
dt_model <- rpart(AttritionFlag ~ ., data = balanced_dataset, method = "class")
rpart.plot(dt_model)

# Make predictions on the test set using the trained Decision Tree model
predictions_dt <- predict(dt_model, test_features, type = "class")

# Evaluate the performance of the Decision Tree model
conf_matrix_dt <- confusionMatrix(predictions_dt, test_target)
print(conf_matrix_dt)

# Extract and print various performance metrics
accuracy_dt <- conf_matrix_dt$overall["Accuracy"]
precision_dt <- conf_matrix_dt$byClass["Precision"]
recall_dt <- conf_matrix_dt$byClass["Sensitivity"]
f1_score_dt <- conf_matrix_dt$byClass["F1"]

cat("Decision Tree Model Metrics:\n")
cat("Accuracy:", accuracy_dt, "\n")
cat("Precision:", precision_dt, "\n")
cat("Recall (Sensitivity):", recall_dt, "\n")
cat("F1 Score:", f1_score_dt, "\n")

title_dt <- sprintf("Confusion Matrix - Decision Tree Model - Accuracy: %.2f%%", accuracy_dt * 100)
title_dt_roc <- sprintf("ROC Curve - Decision Tree Model - Accuracy: %.2f%%", accuracy_dt * 100)
# Plot confusion matrix
show_confusion_matrix(conf_matrix_dt, title = title_dt)
# Plot ROC-AUC Curve
plot_roc_curve(predictions_dt, y_test, title=title_dt_roc)
############################################################################################
#####################[Neural Network with cross-validation]- [SHEILA] ##################
####################################################################################################
# Define the formula
formula <- AttritionFlag ~ .
# Defining the selected features
selected_features <- c("CustomerAge", "Gender","Dependentcount","EducationLevel","IncomeCategory","CardCategory","TotalRelationshipCount","TotalRevolvingBal","TotalTransCt","TotalCtChngQ4Q1" ,"ContactsCount12mon","CreditLimit","AvgOpenToBuy","TotalAmtChngQ4Q1","AvgUtilizationRatio")

# Selecting the features for the training data
X_train <- as.matrix(balanced_dataset[selected_features])
y_train <- as.numeric(balanced_dataset$AttritionFlag)

# Selecting the features for the testing data
X_test <- as.matrix(testing_data[selected_features])
y_test <- as.numeric(testing_data$AttritionFlag)

# Convert y_train to a factor with two levels
y_train <- factor(y_train, levels = c(0, 1))

# Create a Neural Network using cross-validation approach
model <- train(
  x = X_train,
  y = y_train,
  method = "nnet",
  trControl = trainControl(method = "cv"),
  preProcess = c("center", "scale")
)

# Make predictions
predictions_nn <- predict(model, newdata = X_test)

# Evaluate performance
conf_matrix_nn <- confusionMatrix(predictions_nn, factor(y_test))
print(conf_matrix_nn)

# Extracting various performance metrics
accuracy_nn <- conf_matrix_nn$overall["Accuracy"]
precision_nn <- conf_matrix_nn$byClass["Precision"]
recall_nn <- conf_matrix_nn$byClass["Sensitivity"]
f1_score_nn <- conf_matrix_nn$byClass["F1"]

# Print the metrics for the neural network model
cat("Accuracy:", accuracy_nn, "\n")
cat("Precision:", precision_nn, "\n")
cat("Recall (Sensitivity):", recall_nn, "\n")
cat("F1 Score:", f1_score_nn, "\n")

# Display confusion Matrix for Random Forest
title_nn <- sprintf("Confusion Matrix - Sheila's Neural Network - Accuracy: %.2f%%", accuracy_nn * 100)
title_nn_roc <- sprintf("ROC Curve - Sheila's Neural Network - Accuracy: %.2f%%", accuracy_nn * 100)
show_confusion_matrix(conf_matrix_nn, title = title_nn)
# Plot ROC-AUC Curve
plot_roc_curve(predictions_nn, y_test, title=title_nn_roc)

####################################################################################################
######################## Logistic Regression Model Training and Evaluation - [SHIVA] ###############
####################################################################################################
# Build the Logistic Regression model
log_reg_model <- glm(AttritionFlag ~ ., data = balanced_dataset, family = "binomial")

# Prepare the test set
test_features_lr <- testing_data[, !names(testing_data) %in% "AttritionFlag"]
test_target_lr <- as.factor(testing_data$AttritionFlag)

# Make predictions on the test set using the trained Logistic Regression model
predictions_lr_prob <- predict(log_reg_model, test_features_lr, type = "response")
predictions_lr <- ifelse(predictions_lr_prob > 0.5, 1, 0)
predictions_lr <- factor(predictions_lr, levels = levels(test_target_lr))

# Evaluate the performance of the Logistic Regression model
conf_matrix_lr <- confusionMatrix(predictions_lr, test_target_lr)
print(conf_matrix_lr)

# Extract and print various performance metrics
accuracy_lr <- conf_matrix_lr$overall["Accuracy"]
precision_lr <- conf_matrix_lr$byClass["Precision"]
recall_lr <- conf_matrix_lr$byClass["Sensitivity"]
f1_score_lr <- conf_matrix_lr$byClass["F1"]

cat("Logistic Regression Model Metrics:\n")
cat("Accuracy:", accuracy_lr, "\n")
cat("Precision:", precision_lr, "\n")
cat("Recall (Sensitivity):", recall_lr, "\n")
cat("F1 Score:", f1_score_lr, "\n")

title_lr <- sprintf("Confusion Matrix - Logistic Regression Model - Accuracy: %.2f%%", accuracy_lr * 100)
title_lr_roc <- sprintf("ROC Curve - Logistic Regression Model - Accuracy: %.2f%%", accuracy_lr * 100)
# Plot confusion matrix
show_confusion_matrix(conf_matrix_lr, title = title_lr)
# Plot ROC-AUC Curve
plot_roc_curve(predictions_lr, y_test, title=title_lr_roc)

############################################################################################
####################[[Neural Network with cross-validation]- [SHIVA]] ##################
####################################################################################################
# Define the formula
formula <- AttritionFlag ~ .
# Defining the selected features
selected_features <- c("CustomerAge", "Gender","Dependentcount","EducationLevel","IncomeCategory","CardCategory","TotalRelationshipCount","TotalRevolvingBal","TotalTransCt","TotalCtChngQ4Q1" ,"ContactsCount12mon", "MonthsInactive12mon","CreditLimit")

# Selecting the features for the training data
X_train <- as.matrix(balanced_dataset[selected_features])
y_train <- as.numeric(balanced_dataset$AttritionFlag)

# Selecting the features for the testing data
X_test <- as.matrix(testing_data[selected_features])
y_test <- as.numeric(testing_data$AttritionFlag)

# Convert y_train to a factor with two levels
y_train <- factor(y_train, levels = c(0, 1))

# Create a Neural Network using cross-validation approach
model <- train(
  x = X_train,
  y = y_train,
  method = "nnet",
  trControl = trainControl(method = "cv"),
  preProcess = c("center", "scale")
)

# Make predictions
predictions_nnSv <- predict(model, newdata = X_test)

# Evaluate performance
conf_matrix_nnSv <- confusionMatrix(predictions_nnSv, factor(y_test))
print(conf_matrix_nnSv)

# Extracting various performance metrics
accuracy_nnSv <- conf_matrix_nnSv$overall["Accuracy"]
precision_nnSv <- conf_matrix_nnSv$byClass["Precision"]
recall_nnSv <- conf_matrix_nnSv$byClass["Sensitivity"]
f1_score_nnSv <- conf_matrix_nnSv$byClass["F1"]

# Print the metrics for the neural network model
cat("Accuracy:", accuracy_nnSv, "\n")
cat("Precision:", precision_nnSv, "\n")
cat("Recall (Sensitivity):", recall_nnSv, "\n")
cat("F1 Score:", f1_score_nnSv, "\n")

# Display confusion Matrix for Random Forest
title_nn <- sprintf("Confusion Matrix - Shiva's Neural Network - Accuracy: %.2f%%", accuracy_nn * 100)
title_nn_roc <- sprintf("ROC Curve - Shiva's Neural Network - Accuracy: %.2f%%", accuracy_nn * 100)
show_confusion_matrix(conf_matrix_nnSv, title = title_nn)
# Plot ROC-AUC Curve
plot_roc_curve(predictions_nnSv, y_test, title=title_nn_roc)

############################################################################################
############################[SHUBHAM's Code - START]########################################
##############################[XGBOOST - [SHUBHAM]]#########################################
############################################################################################
# Convert data frames to matrices
X_train <- as.matrix(balanced_dataset[, !names(balanced_dataset) %in% "AttritionFlag"])
y_train <- as.numeric(balanced_dataset$AttritionFlag)

X_test <- as.matrix(testing_data[, !names(testing_data) %in% "AttritionFlag"])
y_test <- as.numeric(testing_data$AttritionFlag)
# Define the formula
formula <- AttritionFlag ~ .

# Ensure the column names are the same in X_train and X_test
colnames(X_test) <- colnames(X_train)

# Train the XGBoost model
model <- xgboost(data = X_train, label = y_train, nrounds = 100, print_every_n = 10)

# Make predictions on the test set
predictions <- predict(model, newdata = X_test)

# Convert predicted probabilities to binary predictions
binary_predictions <- factor(ifelse(predictions > 0.5, 1, 0), levels = levels(factor(y_test)))

# Evaluate performance
conf_matrix_xgboost <- confusionMatrix(binary_predictions, factor(y_test))
print(conf_matrix)

# Extracting various performance metrics
accuracy_xgb <- conf_matrix_xgboost$overall["Accuracy"]
precision_xgb <- conf_matrix_xgboost$byClass["Precision"]
recall_xgb <- conf_matrix_xgboost$byClass["Sensitivity"]
f1_score_xgb <- conf_matrix_xgboost$byClass["F1"]

# Print the metrics for the balanced model
cat("XGBoost Model Metrics:\n")
cat("Accuracy:", accuracy_xgb, "\n")
cat("Precision:", precision_xgb, "\n")
cat("Recall (Sensitivity):", recall_xgb, "\n")
cat("F1 Score:", f1_score_xgb, "\n")
# Create a title string including the accuracy
title_xgb <- sprintf("Confusion Matrix - XGBoost Model - Accuracy: %.2f%%", accuracy_xgb * 100)
title_xgb_roc <- sprintf("ROC Curve - XGBoost Model - Accuracy: %.2f%%", accuracy_xgb * 100)
# Plot confusion matrix
show_confusion_matrix(conf_matrix_xgboost, title=title_xgb)
# Plot ROC-AUC Curve
plot_roc_curve(predictions, y_test, title=title_xgb_roc)
###################################### END ################################################
###########################################################################################
########################[XGBOOST with Cross-Validation- [SHUBHAM]]#########################
# Define the formula
formula <- AttritionFlag ~ .

# Convert data frames to matrices
X <- as.matrix(balanced_dataset[, !names(balanced_dataset) %in% "AttritionFlag"])
y <- as.numeric(balanced_dataset$AttritionFlag)

# Create a data matrix
dtrain <- xgb.DMatrix(data = X, label = y)

# Set up cross-validation parameters
cv_params <- list(
  objective = "binary:logistic", 
  eval_metric = "logloss"
)

# Perform cross-validation using xgb.cv
cv_model <- xgb.cv(
  params = cv_params,
  data = dtrain,
  nfold = 5,
  nrounds = 200,
  early_stopping_rounds = 10,
  maximize = FALSE,
  verbose = TRUE
)

# Print the cross-validation results
print(cv_model)

# Find the optimal number of rounds based on cross-validation
best_nrounds <- cv_model$best_iteration

# Train the final XGBoost model with the optimal number of rounds
final_model <- xgboost(data = dtrain, nrounds = best_nrounds, params = cv_params, verbose = TRUE)

# Make predictions on the test set
predictions <- predict(final_model, newdata = X_test)

# Convert predicted probabilities to binary predictions
binary_predictions <- factor(ifelse(predictions > 0.5, 1, 0), levels = levels(factor(y_test)))

# Evaluate performance
conf_matrix_xgb_cv <- confusionMatrix(binary_predictions, factor(y_test))
print(conf_matrix_xgb_cv)

# Extracting various performance metrics
accuracy_xgb_cv <- conf_matrix_xgb_cv$overall["Accuracy"]
precision_xgb_cv <- conf_matrix_xgb_cv$byClass["Precision"]
recall_xgb_cv <- conf_matrix_xgb_cv$byClass["Sensitivity"]
f1_score_xgb_cv <- conf_matrix_xgb_cv$byClass["F1"]

# Print the metrics for the final model
cat("Accuracy:", accuracy_xgb_cv, "\n")
cat("Precision:", precision_xgb_cv, "\n")
cat("Recall (Sensitivity):", recall_xgb_cv, "\n")
cat("F1 Score:", f1_score_xgb_cv, "\n")

# Create a title string including the accuracy
title_xgb_cv <- sprintf("Confusion Matrix xgBoost after Cross-Validation - Accuracy: %.2f%%", accuracy_xgb_cv * 100)
title_xgb_cv_roc <- sprintf("ROC Curve - XGBoost after Cross-Validation - Accuracy: %.2f%%", accuracy_xgb_cv * 100)
# Plot confusion matrix
show_confusion_matrix(conf_matrix_xgb_cv, title = title_xgb_cv)
# Plot ROC-AUC Curve
plot_roc_curve(predictions, y_test, title=title_xgb_cv_roc)
#####################################[END]#################################################
###########################################################################################
#####################[[Neural Network with cross-validation]- [SHUBHAM]] ##################
# Define the formula
formula <- AttritionFlag ~ .

# Convert y_train to a factor with two levels
y_train <- factor(y_train, levels = c(0, 1))

# Create a Neural Network using cross-validation approach
model <- train(
  x = X_train,
  y = y_train,
  method = "nnet",
  trControl = trainControl(method = "cv"),
  preProcess = c("center", "scale")
)

# Make predictions
predictions_nn_shu <- predict(model, newdata = X_test)

# Evaluate performance
conf_matrix_nn_shu <- confusionMatrix(predictions_nn_shu, factor(y_test))
print(conf_matrix_nn_shu)

# Extracting various performance metrics
accuracy_nn_shu <- conf_matrix_nn_shu$overall["Accuracy"]
precision_nn_shu <- conf_matrix_nn_shu$byClass["Precision"]
recall_nn_shu <- conf_matrix_nn_shu$byClass["Sensitivity"]
f1_score_nn_shu <- conf_matrix_nn_shu$byClass["F1"]

# Print the metrics for the neural network model
cat("Accuracy:", accuracy_nn_shu, "\n")
cat("Precision:", precision_nn_shu, "\n")
cat("Recall (Sensitivity):", recall_nn_shu, "\n")
cat("F1 Score:", f1_score_nn_shu, "\n")

# Display confusion Matrix for Random Forest
title_nn <- sprintf("Confusion Matrix - Shubham's Neural Network - Accuracy: %.2f%%", accuracy_nn_shu * 100)
title_nn_roc <- sprintf("ROC Curve - Shubham's Neural Network - Accuracy: %.2f%%", accuracy_nn_shu * 100)
# Plot confusion matrix
show_confusion_matrix(conf_matrix_nn_shu, title = title_nn)
# Plot ROC-AUC Curve
plot_roc_curve(predictions_nn_shu, y_test, title=title_nn_roc)

# compare my model results
shu_model_metrics <- list()

shu_nn_metrics <- data.frame(
  Model = "Shubham Neural Network",
  Accuracy = accuracy_nn_shu,
  Precision = precision_nn_shu,
  Recall = recall_nn_shu,
  F1_Score = f1_score_nn_shu
)
shu_model_metrics <- append(shu_model_metrics, list(shu_nn_metrics))

shu_xg_metrics <- data.frame(
  Model = "XGBOOST",
  Accuracy = accuracy_xgb,
  Precision = precision_xgb,
  Recall = recall_xgb,
  F1_Score = f1_score_xgb
)
shu_model_metrics <- append(shu_model_metrics, list(shu_xg_metrics))

shu_xcv_metrics <- data.frame(
  Model = "XGBOOST CV",
  Accuracy = accuracy_xgb_cv,
  Precision = precision_xgb_cv,
  Recall = recall_xgb_cv,
  F1_Score = f1_score_xgb_cv
)
shu_model_metrics <- append(shu_model_metrics, list(shu_xcv_metrics))

shu_model_metrics_a <- do.call(rbind, shu_model_metrics)
# Change data for bar chart to show performance comparison
shu_data_long <- tidyr::gather(shu_model_metrics_a, Metric, Value, -Model)

model_chart_title = "ML Model Comparison"
#plot chart
plot_bar_compare(shu_data_long, model_chart_title)

####################################[END]###################################################
############################[SHUBHAM's Code - END]##########################################

#############################[NAVIN's Code - START]########################################
################################ Random Forest [NAVIN] #####################################

# Split the data into features and target
features_rf <- balanced_dataset[, !names(balanced_dataset) %in% "AttritionFlag"]
target_rf <- as.factor(balanced_dataset$AttritionFlag)

# Train the Random Forest model
rf_model <- randomForest(features_rf, target_rf, ntree = 100, importance = TRUE)

# Prepare test set
test_features_rf <- testing_data[, !names(testing_data) %in% "AttritionFlag"]
test_target_rf <- as.factor(testing_data$AttritionFlag)

# Make predictions on the test set
predictions_rf <- predict(rf_model, newdata = test_features_rf)

# Evaluate performance
conf_matrix_rf <- confusionMatrix(predictions_rf, test_target_rf)
print(conf_matrix_rf)

# Extracting various performance metrics
accuracy_rf <- conf_matrix_rf$overall["Accuracy"]
precision_rf <- conf_matrix_rf$byClass["Precision"]
recall_rf <- conf_matrix_rf$byClass["Sensitivity"]
f1_score_rf <- conf_matrix_rf$byClass["F1"]

# Print the metrics for the Random Forest model
cat("Random Forest Model Metrics:\n")
cat("Accuracy:", accuracy_rf, "\n")
cat("Precision:", precision_rf, "\n")
cat("Recall (Sensitivity):", recall_rf, "\n")
cat("F1 Score:", f1_score_rf, "\n")

# Display confusion Matrix for Random Forest
title_rf <- sprintf("Confusion Matrix - Random Forest Model - Accuracy: %.2f%%", accuracy_rf * 100)
title_rf_roc <- sprintf("ROC Curve - Random Forest Model - Accuracy: %.2f%%", accuracy_rf * 100)
# Plot confusion matrix
show_confusion_matrix(conf_matrix_rf, title = title_rf)
# Plot ROC-AUC Curve
plot_roc_curve(predictions_rf, y_test, title=title_rf_roc)
######################################### END Random Forest#################################
############################################################################################
######################## Random Forest with Cross Validation [START] #######################
# prepare the data using features and target 
myRf_data <- data.frame(features_rf, target_rf)

# Rf Control parameters for cross validation with 5 fold
ctrl <- trainControl(method = "cv", number = 5)

# Train the Random Forest model with cross-validation
rf_model_cv <- train(
  target_rf ~ ., 
  data = myRf_data, 
  method = "rf", 
  ntree = 100, 
  importance = TRUE,
  trControl = ctrl
)

# Print the cross-validated results
print(rf_model_cv)

# Prepare test set
test_features_rf <- testing_data[, !names(testing_data) %in% "AttritionFlag"]
test_target_rf <- as.factor(testing_data$AttritionFlag)

# Make predictions on the test set
predictions_rfcv <- predict(rf_model_cv, newdata = test_features_rf)

# Evaluate performance
conf_matrix_rfcv <- confusionMatrix(predictions_rfcv, test_target_rf)
print(conf_matrix_rfcv)

# Extracting various performance metrics
accuracy_rfcv <- conf_matrix_rfcv$overall["Accuracy"]
precision_rfcv <- conf_matrix_rfcv$byClass["Precision"]
recall_rfcv <- conf_matrix_rfcv$byClass["Sensitivity"]
f1_score_rfcv <- conf_matrix_rfcv$byClass["F1"]

# Print the metrics for the Random Forest model with
cat("Random Forest Model after Cross validation Metrics:\n")
cat("Accuracy:", accuracy_rfcv, "\n")
cat("Precision:", precision_rfcv, "\n")
cat("Recall (Sensitivity):", recall_rfcv, "\n")
cat("F1 Score:", f1_score_rfcv, "\n")

# Display confusion Matrix for Random Forest
title_rfcv <- sprintf("Confusion Matrix - Random Forest Model with Cross validation - Accuracy: %.2f%%", accuracy_rfcv * 100)
title_rfcv_roc <- sprintf("ROC Curve - Random Forest Model Cross validation - Accuracy: %.2f%%", accuracy_rfcv * 100)
# Plot confusion matrix
show_confusion_matrix(conf_matrix_rfcv, title = title_rfcv)
# Plot ROC-AUC Curve
plot_roc_curve(predictions_rfcv, y_test, title=title_rfcv_roc)

####################### [Random Forest with Cross Validation [END]] #####################
#########################################################################################
#########################[[Neural Network common Model - [NAVIN]] #######################
# Define the formula
formula <- AttritionFlag ~ .

# Convert y_train to a factor with two levels
y_train <- factor(y_train, levels = c(0, 1))

# Create a Neural Network model with the nnet method
navin_nn_model <- train(
  x = X_train,
  y = y_train,
  method = "nnet",
  trControl = trainControl(
    method = "repeatedcv",   # Repeated Cross-Validation method
    number = 5,              # Number of folds is 5
    repeats = 3              # Number of repetitions is 3
  ),
  preProcess = c("center", "scale"),
  tuneGrid = expand.grid(size = c(5, 10, 15), decay = c(0.1, 0.01, 0.001)),
  metric = "Accuracy",
  maximize = TRUE
)

# Make predictions
predictions_navinn <- predict(navin_nn_model, newdata = X_test)

# Evaluate performance
conf_matrix_navinn <- confusionMatrix(predictions_navinn, factor(y_test))
print(conf_matrix_navinn)

#####################################
# Extracting various performance metrics
accuracy_navinn <- conf_matrix_navinn$overall["Accuracy"]
precision_navinn <- conf_matrix_navinn$byClass["Precision"]
recall_navinn <- conf_matrix_navinn$byClass["Sensitivity"]
f1_score_navinn <- conf_matrix_navinn$byClass["F1"]

# Print the metrics for the neural network model
cat("Accuracy:", accuracy_navinn, "\n")
cat("Precision:", precision_navinn, "\n")
cat("Recall (Sensitivity):", recall_navinn, "\n")
cat("F1 Score:", f1_score_navinn, "\n")

# Display confusion Matrix for Random Forest
title_navinn <- sprintf("Confusion Matrix - Navin's Neural Network - Accuracy: %.2f%%", accuracy_navinn * 100)
title_navinn_roc <- sprintf("ROC Curve - Navin's Neural Network - Accuracy: %.2f%%", accuracy_navinn * 100)
# Plot confusion matrix
show_confusion_matrix(conf_matrix_navinn, title = title_navinn)
# Plot ROC-AUC Curve
plot_roc_curve(predictions_navinn, y_test, title=title_navinn_roc)

#########################[NAVIN's Code - END Neural Network]##############################
##########################################################################################
######## Comparing the result of Navin's ML models algorithms ##############
navi_model_metrics <- list()
# Random Forest Model Metrics Collection
rf_metrics <- data.frame(
  Model = "Random Forest",
  Accuracy = accuracy_rf,
  Precision = precision_rf,
  Recall = recall_rf,
  F1_Score = f1_score_rf
)
navi_model_metrics <- append(navi_model_metrics, list(rf_metrics))

# Random Forest Model with Cross validation  Metrics Collection
rfcv_metrics <- data.frame(
  Model = "Random Forest CV",
  Accuracy = accuracy_rfcv,
  Precision = precision_rfcv,
  Recall = recall_rfcv,
  F1_Score = f1_score_rfcv
)
navi_model_metrics <- append(navi_model_metrics, list(rfcv_metrics))

# Navin Neural Network Metrics Collection
navi_nn_metrics <- data.frame(
  Model = "Navin Neural Network",
  Accuracy = accuracy_navinn,
  Precision = precision_navinn,
  Recall = recall_navinn,
  F1_Score = f1_score_navinn
)
navi_model_metrics <- append(navi_model_metrics, list(navi_nn_metrics))

# Combine all metrics into one data frame
navi_model_metrics <- do.call(rbind, navi_model_metrics)

# Create a table to show comparison
navi_model_metrics_table <- kable(navi_model_metrics, "html") %>%
  kable_styling(bootstrap_options = "striped", full_width = FALSE)

#display the table
print(navi_model_metrics_table)
# display the table
print(navi_model_metrics)

# Change data for bar chart to show performance comparison
navi_data_long <- tidyr::gather(navi_model_metrics, Metric, Value, -Model)
navi_chart_title = "Navin's ML Model Comparison"

#plot chart
plot_bar_compare(navi_data_long, navi_chart_title)

#########################################################################################
##################################[NAVIN's Code - END ]##################################
########################### Group Model Result comparison ###############################
common_model_metrics <- list()

fin_nn_metrics <- data.frame(
  Model = "Fin",
  Accuracy = accuracy_nn_FIN,
  Precision = precision_nn_FIN,
  Recall = recall_nn_FIN,
  F1_Score = f1_score_nn_FIN
)
common_model_metrics <- append(common_model_metrics, list(fin_nn_metrics))


shl_nn_metrics <- data.frame(
  Model = "Sheila",
  Accuracy = accuracy_nn,
  Precision = precision_nn,
  Recall = recall_nn,
  F1_Score = f1_score_nn
)
common_model_metrics <- append(common_model_metrics, list(shl_nn_metrics))


shv_nn_metrics <- data.frame(
  Model = "Shiva",
  Accuracy = accuracy_nnSv,
  Precision = precision_nnSv,
  Recall = recall_nnSv,
  F1_Score = f1_score_nnSv
)
common_model_metrics <- append(common_model_metrics, list(shv_nn_metrics))

shu_nn_metrics <- data.frame(
  Model = "Shubham",
  Accuracy = accuracy_nn_shu,
  Precision = precision_nn_shu,
  Recall = recall_nn_shu,
  F1_Score = f1_score_nn_shu
)
common_model_metrics <- append(common_model_metrics, list(shu_nn_metrics))

Navi_nn_metrics <- data.frame(
  Model = "Navin",
  Accuracy = accuracy_navinn,
  Precision = precision_navinn,
  Recall = recall_navinn,
  F1_Score = f1_score_navinn
)
common_model_metrics <- append(common_model_metrics, list(Navi_nn_metrics))

#################################################################################
# Combine all metrics into one data frame
common_model_metrics <- do.call(rbind, common_model_metrics)
# Create a table to show comparison
common_model_metrics_table <- kable(common_model_metrics, "html") %>%
  kable_styling(bootstrap_options = "striped", full_width = FALSE)
#display the table
print(common_model_metrics_table)

# display the table
print(common_model_metrics)
# Change data for bar chart to show performance comparison
common_data_long <- tidyr::gather(common_model_metrics, Metric, Value, -Model)
common_chart_title = "Neural Networkl ML Model Comparison"

#plot chart
plot_bar_compare(common_data_long, common_chart_title)

# Initialize a list to store all model metrics
all_model_metrics <- list()

# SVM Model Metrics Collection
svm_metrics <- data.frame(
  Model = "SVM",
  Accuracy = accuracy_svm,
  Precision = precision_svm,
  Recall = recall_svm,
  F1_Score = f1_score_svm
)
all_model_metrics <- append(all_model_metrics, list(svm_metrics))

# Decision Tree Model Metrics Collection
dt_metrics <- data.frame(
  Model = "Decision Tree",
  Accuracy = accuracy_dt,
  Precision = precision_dt,
  Recall = recall_dt,
  F1_Score = f1_score_dt
)
all_model_metrics <- append(all_model_metrics, list(dt_metrics))

# Logistic Regression Model Metrics Collection
lr_metrics <- data.frame(
  Model = "Logistic Regression",
  Accuracy = accuracy_lr,
  Precision = precision_lr,
  Recall = recall_lr,
  F1_Score = f1_score_lr
)
all_model_metrics <- append(all_model_metrics, list(lr_metrics))


# XGBoost Model Metrics Collection
xgboost_metrics <- data.frame(
  Model = "XGBoost",
  Accuracy = accuracy_xgb,
  Precision = precision_xgb,
  Recall = recall_xgb,
  F1_Score = f1_score_xgb
)
all_model_metrics <- append(all_model_metrics, list(xgboost_metrics))

# Random Forest Model Metrics Collection
rf_metrics <- data.frame(
  Model = "Random Forest",
  Accuracy = accuracy_rf,
  Precision = precision_rf,
  Recall = recall_rf,
  F1_Score = f1_score_rf
)
all_model_metrics <- append(all_model_metrics, list(rf_metrics))

# Combine all metrics into one data frame
final_model_metrics <- do.call(rbind, all_model_metrics)

# Create a table to show comparison
final_model_metrics_table <- kable(final_model_metrics, "html") %>%
  kable_styling(bootstrap_options = "striped", full_width = FALSE)

#display the table
print(final_model_metrics_table)

# Change data for bar chart to show performance comparison
data_long <- tidyr::gather(final_model_metrics, Metric, Value, -Model)

model_chart_title = "ML Model Comparison"
#plot chart
plot_bar_compare(navi_data_long, navi_chart_title)

# Write model performance metrics result to CSV file
write.csv(final_model_metrics, "Result_model_performance_metrics.csv", row.names = FALSE)
############# End OF THE CODE ############

print("end")

