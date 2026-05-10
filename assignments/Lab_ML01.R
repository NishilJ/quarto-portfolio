# ============================================================================
# Workshop: Machine Learning with Survey Data (TEDS 2016)
# EPPS 6323 Knowledge Mining
# Karl Ho, University of Texas at Dallas
# ============================================================================

# ----------------------------------------------------------------------------
# 0. Setup: Packages and Data
# ----------------------------------------------------------------------------

library(haven)        # Read Stata files
library(tidyverse)    # Data wrangling and visualization
library(GGally)       # Pairs plots
library(cluster)      # Clustering
library(factoextra)   # Visualize clusters and PCA
library(rpart)        # Decision trees
library(rpart.plot)   # Plot decision trees
library(randomForest) # Random forest
library(caret)        # Model training and evaluation
library(e1071)        # SVM and Naive Bayes

# Load data
TEDS_2016 <- read_stata("https://github.com/datageneration/home/blob/master/DataProgramming/data/TEDS_2016.dta?raw=true")


# ============================================================================
# PART I: EXPLORATORY DATA ANALYSIS
# ============================================================================

# ----------------------------------------------------------------------------
# Q1. Data overview
# ----------------------------------------------------------------------------

dim(TEDS_2016)
names(TEDS_2016)
summary(TEDS_2016)

# ----------------------------------------------------------------------------
# Q2. Missing data assessment
# ----------------------------------------------------------------------------

TEDS_2016 %>%
  summarise(across(everything(), ~sum(is.na(.)))) %>%
  pivot_longer(everything(), 
               names_to = "variable", 
               values_to = "n_missing") %>%
  filter(n_missing > 0) %>%
  arrange(desc(n_missing))

# Visualize missingness
TEDS_2016 %>%
  summarise(across(everything(), ~sum(is.na(.)))) %>%
  pivot_longer(everything(), 
               names_to = "variable", 
               values_to = "n_missing") %>%
  filter(n_missing > 0) %>%
  ggplot(aes(x = reorder(variable, n_missing), y = n_missing)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "Missing Values by Variable",
       x = NULL, y = "Count of Missing Values") +
  theme_minimal(base_size = 18)

# ----------------------------------------------------------------------------
# Q3. Variable recoding
# ----------------------------------------------------------------------------

teds <- TEDS_2016 %>%
  mutate(
    vote = factor(votetsai, 
                  levels = c(0, 1), 
                  labels = c("Other", "Tsai")),
    gender = factor(female, 
                    levels = c(0, 1), 
                    labels = c("Male", "Female")),
    Tondu = as.factor(Tondu),
    Party = as.factor(Party)
  ) %>%
  dplyr::select(vote, gender, age, edu, income, 
         Taiwanese, Econ_worse, Tondu, Party, DPP) %>%
  drop_na()

glimpse(teds)

# ----------------------------------------------------------------------------
# Q4. Vote choice distribution
# ----------------------------------------------------------------------------

ggplot(teds, aes(x = vote, fill = vote)) +
  geom_bar() +
  scale_fill_manual(values = c("Other" = "#2980b9", "Tsai" = "#27ae60")) +
  labs(title = "Vote Choice Distribution",
       subtitle = "TEDS 2016: Tsai Ing-wen vs. Other Candidates",
       x = "Vote Choice", y = "Count") +
  theme_minimal(base_size = 18) +
  theme(legend.position = "none")

# ----------------------------------------------------------------------------
# Q5. Age distribution
# ----------------------------------------------------------------------------

ggplot(teds, aes(x = age)) +
  geom_histogram(binwidth = 5, fill = "steelblue", color = "white") +
  labs(title = "Age Distribution of Respondents",
       x = "Age", y = "Count") +
  theme_minimal(base_size = 18)

# ----------------------------------------------------------------------------
# Q6. Vote by gender
# ----------------------------------------------------------------------------

ggplot(teds, aes(x = gender, fill = vote)) +
  geom_bar(position = "fill") +
  scale_fill_manual(values = c("Other" = "#2980b9", "Tsai" = "#27ae60")) +
  labs(title = "Vote Choice by Gender",
       x = "Gender", y = "Proportion",
       fill = "Vote") +
  theme_minimal(base_size = 18)

# ----------------------------------------------------------------------------
# Q7. Age by vote choice
# ----------------------------------------------------------------------------

ggplot(teds, aes(x = vote, y = age, fill = vote)) +
  geom_boxplot() +
  scale_fill_manual(values = c("Other" = "#2980b9", "Tsai" = "#27ae60")) +
  labs(title = "Age Distribution by Vote Choice",
       x = "Vote Choice", y = "Age") +
  theme_minimal(base_size = 18) +
  theme(legend.position = "none")

# ----------------------------------------------------------------------------
# Q8. Correlation matrix
# ----------------------------------------------------------------------------

teds %>%
  dplyr::select(age, edu, income, Taiwanese, Econ_worse, DPP) %>%
  cor(use = "complete.obs") %>%
  as.data.frame() %>%
  rownames_to_column("var1") %>%
  pivot_longer(-var1, names_to = "var2", values_to = "cor") %>%
  ggplot(aes(x = var1, y = var2, fill = cor)) +
  geom_tile() +
  geom_text(aes(label = round(cor, 2)), size = 6) +
  scale_fill_gradient2(low = "#e74c3c", mid = "white", high = "#2980b9",
                       midpoint = 0, limits = c(-1, 1)) +
  labs(title = "Correlation Matrix of Numeric Variables",
       x = NULL, y = NULL) +
  theme_minimal(base_size = 18) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# ----------------------------------------------------------------------------
# Q9. Pairs plot
# ----------------------------------------------------------------------------

teds %>%
  dplyr::select(age, edu, income, Taiwanese, Econ_worse, vote) %>%
  ggpairs(aes(color = vote, alpha = 0.5),
          upper = list(continuous = wrap("cor", size = 4)),
          lower = list(continuous = wrap("points", size = 0.5))) +
  scale_color_manual(values = c("Other" = "#2980b9", "Tsai" = "#27ae60")) +
  scale_fill_manual(values = c("Other" = "#2980b9", "Tsai" = "#27ae60")) +
  theme_minimal(base_size = 14)


# ============================================================================
# PART II: UNSUPERVISED LEARNING
# ============================================================================

# ----------------------------------------------------------------------------
# Prepare data for clustering
# ----------------------------------------------------------------------------

teds_numeric <- teds %>%
  dplyr::select(age, edu, income, Taiwanese, Econ_worse, DPP) %>%
  scale() %>%
  as.data.frame()

head(teds_numeric)

# ----------------------------------------------------------------------------
# Q10. Choosing k: Elbow method
# ----------------------------------------------------------------------------

fviz_nbclust(teds_numeric, kmeans, method = "wss", k.max = 10) +
  labs(title = "Elbow Method: Optimal Number of Clusters") +
  theme_minimal(base_size = 18)

# Silhouette method
fviz_nbclust(teds_numeric, kmeans, method = "silhouette", k.max = 10) +
  labs(title = "Silhouette Method: Optimal Number of Clusters") +
  theme_minimal(base_size = 18)

# ----------------------------------------------------------------------------
# Q11. K-Means clustering
# ----------------------------------------------------------------------------

set.seed(6323)
km_result <- kmeans(teds_numeric, centers = 3, nstart = 25)

table(km_result$cluster)
round(km_result$centers, 2)

# Visualize clusters
fviz_cluster(km_result, data = teds_numeric,
             geom = "point", 
             ellipse.type = "convex",
             palette = c("#e74c3c", "#2980b9", "#27ae60"),
             ggtheme = theme_minimal(base_size = 18)) +
  labs(title = "K-Means Clustering of TEDS 2016 Voters (k = 3)")

# ----------------------------------------------------------------------------
# Q12. Cluster profiling
# ----------------------------------------------------------------------------

teds$cluster <- as.factor(km_result$cluster)

teds %>%
  group_by(cluster) %>%
  summarise(
    n = n(),
    mean_age = round(mean(age), 1),
    mean_edu = round(mean(edu), 1),
    mean_income = round(mean(income), 1),
    pct_Taiwanese = round(mean(Taiwanese) * 100, 1),
    pct_Econ_worse = round(mean(Econ_worse) * 100, 1),
    mean_DPP = round(mean(DPP), 2),
    pct_Tsai = round(mean(vote == "Tsai") * 100, 1)
  )

# ----------------------------------------------------------------------------
# Q13. Clusters vs. vote choice
# ----------------------------------------------------------------------------

ggplot(teds, aes(x = cluster, fill = vote)) +
  geom_bar(position = "fill") +
  scale_fill_manual(values = c("Other" = "#2980b9", "Tsai" = "#27ae60")) +
  labs(title = "Vote Choice Distribution by Cluster",
       x = "Cluster", y = "Proportion",
       fill = "Vote") +
  theme_minimal(base_size = 18)

# ----------------------------------------------------------------------------
# Q14. Principal Component Analysis
# ----------------------------------------------------------------------------

pca_result <- prcomp(teds_numeric, scale. = TRUE)
summary(pca_result)

# Scree plot
fviz_eig(pca_result, addlabels = TRUE) +
  labs(title = "Scree Plot: Variance Explained by Each PC") +
  theme_minimal(base_size = 18)

# ----------------------------------------------------------------------------
# Q15. PCA variable contributions
# ----------------------------------------------------------------------------

fviz_pca_var(pca_result, 
             col.var = "contrib",
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE) +
  labs(title = "PCA: Variable Contributions") +
  theme_minimal(base_size = 18)

# PCA individuals by vote
fviz_pca_ind(pca_result,
             geom = "point",
             col.ind = teds$vote,
             palette = c("#2980b9", "#27ae60"),
             addEllipses = TRUE,
             legend.title = "Vote") +
  labs(title = "PCA: Voters Projected onto First Two PCs") +
  theme_minimal(base_size = 18)


# ============================================================================
# PART III: SUPERVISED LEARNING
# ============================================================================

# ----------------------------------------------------------------------------
# Q16. Train-test split
# ----------------------------------------------------------------------------

set.seed(6323)

train_index <- createDataPartition(teds$vote, p = 0.7, list = FALSE)
train_data <- teds[train_index, ] %>% dplyr::select(-cluster)
test_data  <- teds[-train_index, ] %>% dplyr::select(-cluster)

cat("Training set:", nrow(train_data), "observations\n")
cat("Test set:", nrow(test_data), "observations\n")
prop.table(table(train_data$vote))

# ----------------------------------------------------------------------------
# Q17. Logistic regression
# ----------------------------------------------------------------------------

logit_model <- glm(vote ~ age + gender + edu + income + 
                     Taiwanese + Econ_worse + DPP,
                   data = train_data, 
                   family = binomial)

summary(logit_model)

# Evaluation
logit_probs <- predict(logit_model, newdata = test_data, type = "response")
logit_pred <- ifelse(logit_probs > 0.5, "Tsai", "Other")
logit_pred <- factor(logit_pred, levels = c("Other", "Tsai"))

confusionMatrix(logit_pred, test_data$vote)

# ----------------------------------------------------------------------------
# Q18. Decision tree
# ----------------------------------------------------------------------------

tree_model <- rpart(vote ~ age + gender + edu + income + 
                      Taiwanese + Econ_worse + DPP,
                    data = train_data,
                    method = "class")

rpart.plot(tree_model, 
           type = 4, 
           extra = 104,
           main = "Decision Tree: Predicting Vote Choice",
           box.palette = "BuGn",
           cex = 1.2)

# Evaluation
tree_pred <- predict(tree_model, newdata = test_data, type = "class")
confusionMatrix(tree_pred, test_data$vote)

# ----------------------------------------------------------------------------
# Q19. Random forest
# ----------------------------------------------------------------------------

set.seed(6323)

rf_model <- randomForest(vote ~ age + gender + edu + income + 
                           Taiwanese + Econ_worse + DPP,
                         data = train_data,
                         ntree = 500,
                         importance = TRUE)

print(rf_model)

# Evaluation
rf_pred <- predict(rf_model, newdata = test_data)
confusionMatrix(rf_pred, test_data$vote)

# ----------------------------------------------------------------------------
# Q20. Variable importance
# ----------------------------------------------------------------------------

importance_df <- as.data.frame(importance(rf_model)) %>%
  rownames_to_column("variable") %>%
  arrange(desc(MeanDecreaseGini))

ggplot(importance_df, aes(x = reorder(variable, MeanDecreaseGini), 
                           y = MeanDecreaseGini)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "Random Forest: Variable Importance",
       subtitle = "Mean Decrease in Gini Index",
       x = NULL, y = "Importance") +
  theme_minimal(base_size = 18)

# ----------------------------------------------------------------------------
# Q21. Model comparison
# ----------------------------------------------------------------------------

results <- tibble(
  Model = c("Logistic Regression", "Decision Tree", "Random Forest"),
  Accuracy = c(
    confusionMatrix(logit_pred, test_data$vote)$overall["Accuracy"],
    confusionMatrix(tree_pred, test_data$vote)$overall["Accuracy"],
    confusionMatrix(rf_pred, test_data$vote)$overall["Accuracy"]
  )
)

results %>% arrange(desc(Accuracy))

# Comparison plot
ggplot(results, aes(x = reorder(Model, Accuracy), y = Accuracy, fill = Model)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = paste0(round(Accuracy * 100, 1), "%")), 
            hjust = -0.1, size = 7) +
  coord_flip() +
  scale_fill_manual(values = c("#e74c3c", "#2980b9", "#27ae60")) +
  scale_y_continuous(limits = c(0, 1), labels = scales::percent) +
  labs(title = "Model Comparison: Prediction Accuracy",
       x = NULL, y = "Accuracy") +
  theme_minimal(base_size = 18) +
  theme(legend.position = "none")
