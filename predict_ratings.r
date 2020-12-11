################################################################################
#                                                                              #
# Prediction of Movie Ratings on the MovieLens 10M Data Set                    #
#                                                                              #
# Author: Eric Gabriel                                                         #
# Date: 10th December 2020                                                     #
#                                                                              #
# This R script is part of the the first project submission of the course      #
# HarvardX PH125.9x "Data Science: Capstone".                                  #
#                                                                              #
# It includes the provided code to load the MovieLens 10M data set and to      #
# create the training data set (edx) as well as the hold-out test set          #
# (validation).                                                                #
# Two classes of models are trained: linear regression and matrix              #
# factorization. For each model, the residual mean squared error (RMSE)        #
# on the hold-out test set is computed.                                        #
#                                                                              #
################################################################################

# Install libraries if they are not already installed
if(!require(data.table)) install.packages("data.table") 
if(!require(tidyverse)) install.packages("tidyverse") 
if(!require(caret)) install.packages("caret")
if(!require(tidyr)) install.packages("recosystem")
if(!require(lubridate)) install.packages("lubridate")
if(!require(ggplot2)) install.packages("ggplot2")

# Load necessary libraries
library(data.table)
library(tidyverse)
library(lubridate)
library(caret)
library(recosystem)

################################################################################
# Download MovieLens 10M data set                                              #
################################################################################

# Note: this process could take a couple of minutes

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- "movielens10m.zip"
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

################################################################################
# Create edx set, validation set (final hold-out test set)                     #
################################################################################

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1,
                                  list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

################################################################################
# Preparation                                                                  #
################################################################################

# This function computes the RMSE between y and y_hat
compute_rmse <- function(y, y_hat){
  sqrt(mean((y - y_hat)^2))
}

# Create data frame for RMSE results
rmses <- c()

################################################################################
# Model 1: Mean rating                                                         #
# y_hat = mean_training                                                        #
################################################################################

# Get mean rating of training data
mean_train <- mean(edx$rating)

# Compute RMSE of y_hat = mean_training
rmses["Mean Rating"] <- compute_rmse(validation$rating, mean_train)

################################################################################
# Model 2: Modelling of movie-specific effect                                  #
# y_hat = mean_training + b_movie                                              #
################################################################################

# Compute estimates of the movie bias on the training set
movie_effect <- edx %>%
  group_by(movieId) %>%
  summarize(b_movie = mean(rating - mean_train))

# Add computed movie bias estimates to the test set and compute y_hat
# according to: Y_{u,i} = \mu_{train} + b_i
y_hat_model2 <- validation %>%
  left_join(movie_effect, by='movieId') %>%
  mutate(y_hat_model2 = mean_train + b_movie) %>%
  .$y_hat_model2

# Compute RMSE of y = validation$rating and y_hat
rmses["Movie effect"] <- compute_rmse(validation$rating, y_hat_model2)

################################################################################
# Model 3: Modelling of movie- and user-specific effects                       #
# y_hat = mean_training + b_movie + b_user                                     #
################################################################################

# Compute estimates of the user bias on the training set
user_effect <- edx %>%
  left_join(movie_effect, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_user = mean(rating - mean_train - b_movie))

# Add computed movie and user bias estimates to the test set and compute y_hat
# according to: Y_{u,i} = \mu_{train} + b_i + b_u
y_hat_model3 <- validation %>%
  left_join(movie_effect, by='movieId') %>%
  left_join(user_effect, by='userId') %>%
  mutate(y_hat_model3 = mean_train + b_movie + b_user) %>%
  .$y_hat_model3

# Compute RMSE of y = validation$rating and y_hat
rmses["Movie and user effect"] <- compute_rmse(validation$rating, y_hat_model3)

################################################################################
# Model 4: Modelling of movie-, user- and time-specific effects                #
# y_hat = mean_training + b_movie + b_user + b_time                            #
################################################################################

# Compute estimates of the time bias on the training set
time_effect <- edx %>%
  mutate(month = round_date(as_datetime(timestamp), unit = "month")) %>%
  left_join(movie_effect, by='movieId') %>%
  left_join(user_effect, by='userId') %>%
  group_by(month) %>%
  summarize(b_month = mean(rating - mean_train - b_movie - b_user))

# Add computed movie, user and time bias estimates to the test set and compute 
# y_hat according to: Y_{u,i} = \mu_{train} + b_i + b_u + b_t
y_hat_model4 <- validation %>%
  mutate(month = round_date(as_datetime(timestamp), unit = "month")) %>%
  left_join(movie_effect, by='movieId') %>%
  left_join(user_effect, by='userId') %>%
  left_join(time_effect, by='month') %>%
  mutate(y_hat_model4 = mean_train + b_movie + b_user + b_month) %>%
  .$y_hat_model4

# Compute RMSE of y = validation$rating and y_hat
rmses["Movie, user and time effect"] <-
  compute_rmse(validation$rating, y_hat_model4)

################################################################################
# Model 4: Modelling of movie-, user-, time- and genre-specific effects        #
# y_hat = mean_training + b_movie + b_user + b_time + b_genre                  #
################################################################################

# Get unique list of genre names
genre_names <- unique(unlist(str_split(edx$genres, "\\|")))

# Compute the mean rating for each individual genre
genre_means <- sapply(genre_names, function(g){
  edx %>%
    filter(str_detect(genres, g)) %>%
    summarise(mean_rating = mean(rating)) %>%
    select(mean_rating)
})

# Create a data frame containing the genre names and mean ratings
genre_details_df <- as.data.frame(genre_names)
genre_details_df$genre_mean <- as.numeric(genre_means)

# Summarize the number of ratings per individual genre in the training set
genre_numbers <- sapply(genre_names, function(g){
  edx %>%
    filter(str_detect(genres, g)) %>%
    summarise(total = n()) %>%
    pull(total)
})
# Show number of rating per genre
genre_numbers

# Compute effect per individual genre
genre_effects <- sapply(genre_names, function(g){
  edx %>%
    filter(str_detect(genres, g)) %>%
    mutate(month = round_date(as_datetime(timestamp), unit = "month")) %>%
    left_join(movie_effect, by='movieId') %>%
    left_join(user_effect, by='userId') %>%
    left_join(time_effect, by='month') %>%
    summarize(b_genre = mean(rating - b_movie - mean_train - b_user - b_month)) %>%
    pull(b_genre)
})

# Compute sum of genre effects (of all assigned genres, respectively) for all
# rows in the validation set
genre_effect_sum <- sapply(seq(1:nrow(validation)), function(i){
  sum(genre_effects[str_split(validation[i,]$genres, "\\|", simplify = TRUE)])
})

# Create extended validation set with b_genre
validation_extended <- validation
validation_extended$b_genre <- genre_effect_sum

# Add computed movie, user, time and genre bias estimates to the test set and
# compute y_hat according to: Y_{u,i} = \mu_{train} + b_i + b_u + b_t + b_g
y_hat_model5 <- validation_extended %>% 
  mutate(month = round_date(as_datetime(timestamp), unit = "month")) %>%
  left_join(movie_effect, by = "movieId") %>%
  left_join(user_effect, by = "userId") %>%
  left_join(time_effect, by = "month") %>%
  mutate(prediction = mean_train + b_movie + b_user + b_month + b_genre) %>%
  .$prediction

# Compute RMSE of y = validation$rating and y_hat
rmses["Movie, user, time and genre effect"] <-
  compute_rmse(y_hat_model5, validation$rating)

######################
# ADD REGULARIZATION #
######################

# Create the sequence of values for lambda
lambdas <- seq(2, 10, 0.25)

# Set the seed before creating training/validation data splits of the training
# data to get reproducible results (for versions of R > 3.5)
set.seed(2020)
 
# Create 10 training/validation sets (90%/10%) from the training data for CV
cv_folds <- createDataPartition(edx$rating, times = 10, p = 0.9)

# For each lambda, find b_i & b_u, followed by prediction & testing
# ATTENTION: the code below takes some time 
cv_results <- sapply(lambdas, function(l){
  
  rmse_results <- sapply(cv_folds, function(train_indices){
    # Fill training and validation data sets from the training data for current
    # cross-validation fold
    cv_train <- edx[train_indices,]
    temp <- edx[-train_indices,]
     
    # Make sure userId and movieId in validation set are also in training set
    cv_test <- temp %>% 
    semi_join(cv_train, by = "movieId") %>%
    semi_join(cv_train, by = "userId")
  
    # Add rows removed from validation set back into training set
    removed <- anti_join(temp, cv_test)
    cv_train <- rbind(cv_train, removed)
  
    # Compute mean rating of the training set of the current CV fold
    mean_train <- mean(cv_train$rating)
  
    # Estimate b_i with regularization using current lambda
    b_movie <- cv_train %>%
      group_by(movieId) %>%
      summarize(b_movie = sum(rating - mean_train)/(n()+l))
  
    # Estimate b_u with regularization using current lambda
    b_user <- cv_train %>%
      left_join(b_movie, by="movieId") %>%
      group_by(userId) %>%
      summarize(b_user = sum(rating - b_movie - mean_train)/(n()+l))
  
    # Compute y_hat of validation set of current CV fold
    y_hat <-
      cv_test %>%
      left_join(b_movie, by = "movieId") %>%
      left_join(b_user, by = "userId") %>%
      mutate(prediction = mean_train + b_movie + b_user) %>%
      .$prediction
  
    # Compute the RMSE between y_hat and actual ratings
    compute_rmse(y_hat, cv_test$rating)
    })
  
  # Return mean RMSE of the 10 CV folds for current lambda
  mean(rmse_results)

})

# Select best lambda based on minimum RMSE
best_lambda <- lambdas[which.min(cv_results)]

# Plot mean RMSE of 10-fold CV for each value of lambda
qplot(lambdas, cv_results)

# Reset variable mean_train to overall mean rating of edx training set
mean_train <- mean(edx$rating)

# Recompute estimates of the movie bias on the training set using best lambda
b_movie <- edx %>%
  group_by(movieId) %>%
  summarize(b_movie = sum(rating - mean_train)/(n()+best_lambda))

# Recompute estimates of the user bias on the training set using best lambda
b_user <- edx %>%
  left_join(b_movie, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_user = sum(rating - b_movie - mean_train)/(n()+best_lambda))

# Add (re)computed movie, user, time and genre bias estimates to the test set
# and compute y_hat according to: Y_{u,i} = \mu_{train} + b_i + b_u + b_t + b_g
y_hat_model6 <-
  validation_extended %>%
  mutate(month = round_date(as_datetime(timestamp), unit = "month")) %>%
  left_join(b_movie, by = "movieId") %>%
  left_join(b_user, by = "userId") %>%
  left_join(time_effect, by = "month") %>%
  mutate(prediction = mean_train + b_movie + b_user + b_month + b_genre) %>%
  .$prediction

# Compute RMSE of y = validation$rating and y_hat
rmses["Movie, user, time and genre effect with regularization"] <- 
  compute_rmse(y_hat_model6, validation$rating)

# Cut-off predictions outside of the target range
y_hat_model6_cutoff <- ifelse(y_hat_model6 < 0.5, 0.5, y_hat_model6)
y_hat_model6_cutoff <- ifelse(y_hat_model6_cutoff > 5, 5, y_hat_model6_cutoff)

# Compute RMSE of y = validation$rating and y_hat
rmses["Movie, user, time and genre effect with regularization and cutoff"] <-
  compute_rmse(y_hat_model6_cutoff, validation$rating)

########################
# MATRIX FACTORIZATION #
########################

# Create Reco model
reco = Reco()

# Create training data from edx training set (from memory)
training_data <- data_memory(user_index = edx$userId,
                             item_index = edx$movieId,
                             rating = edx$rating)

# Find best options for training (ATTENTION: this takes a while)
opts = reco$tune(training_data, opts = list(dim = seq(5, 30, 5),
                                            lrate = seq(0.05, 0.3, 0.05),
                                            costp_l1 = 0,
                                            costq_l1 = 0,
                                            nthread = 4,
                                            niter = 20))
# Save best training options
best_tune <- opts$min

# Train Reco model using the training data
reco$train(training_data, opts = c(best_tune, nthread = 1, niter = 20))

# Create test data from hold-out test set (from memory)
test_data <- data_memory(user_index = validation$userId,
                         item_index = validation$movieId,
                         rating = validation$rating)

# Make predictions for the test set using the trained Reco model
predictions <- reco$predict(test_data = test_data, out_pred = out_memory())

# Compute RMSE of y = validation$rating and y_hat
rmses["Matrix Factorization"] <- compute_rmse(validation$rating, predictions)

# Cut-off predictions outside of the target range
predictions_cutoff <- ifelse(predictions < 0.5, 0.5, predictions)
predictions_cutoff <- ifelse(predictions_cutoff > 5, 5, predictions_cutoff)

# Compute RMSE of y = validation$rating and y_hat
rmses["Matrix Factorization with cutoff"] <-
  compute_rmse(validation$rating, predictions_cutoff)