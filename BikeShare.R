# Load Libraries
library(tidyverse)
library(tidymodels)
library(glmnet)
library(vroom)
library(DataExplorer)
library(patchwork)
library(poissonreg)
library(glmnet)

# Read in Data
sampleSubmission <- vroom("sampleSubmission.csv")
test <- vroom("test.csv")
train <- vroom("train.csv")

# Perform EDA
glimpse(train)
plot_intro(train)
plot_correlation(train)
plot_bar(train)
plot_histogram(train)
plot_missing(train)

# Weather Barplot
weather <- ggplot(data = train, mapping = aes(x = weather, y = count)) + 
  geom_col(fill = "blue") +
  labs(x = "Weather", y = "Count", 
       title = "Count of Bicylces 
       Rented Based on Weather")


# Scatterplot of Count vs aTemp (Temp it feels like)
atemp <- ggplot(data = train, mapping = aes(x = atemp, y = count)) +
  geom_point() +
  labs(x = "Temperature It Feels Like (in Celcius)", y = "Count", 
       title = "Count of Bicylces Rented Based on 
       Temperature It Feels Like")

# Scatterplot of Temp vs aTemp
temps <- ggplot(data = train, mapping = aes(x = atemp, y = temp)) +
  geom_point() +
  labs(x = "Temperature It Feels Like (in Celcius)", y = "Actual Temperature(in Celcius)", 
       title = "Count Based on Temp It Feels Like 
       vs Actual Temp")

# Barchart of Count vs Holiday
train$holiday <- factor(train$holiday)
holiday <- ggplot(data = train, mapping = aes(x = holiday, y = count)) + 
  geom_col(fill = "blue") +
  labs(x = "Holiday", y = "Count", 
       title = "Count of Bicylces Rented 
       Based on Holiday Status") +
  scale_x_discrete(labels = c("No", "Yes") )

# Patch them together 
(weather + holiday) / (atemp + temps)

# Make weather 4 observation to weather 3 and make category called "rain"
# notice only 1 obersvation in train 2
train2 <- train |> filter(weather == 4)

# change season numbers into category not number

# Linear Model

# Make workingday a factor
train$workingday <- factor(train$workingday)

# Make log(count) and select variables
train2 <- train |> mutate("log_count"= log(count)) |> select("datetime", "season",
                                                           "holiday", "workingday",
                                                           "weather", "temp", "atemp",
                                                           "humidity", "windspeed", "log_count")

my_linear_model <- linear_reg() |>
  set_engine("lm") |>
  set_mode("regression") |> # means dealing with quantative, numeric target
  fit(formula = log_count ~. , data = train2)

# Generate Preditions using linear model
test$holiday <- factor(test$holiday)
test$workingday <- factor(test$workingday)
bike_predictions <- predict(my_linear_model,
                            new_data = test)
bike_predictions

# Format the Predictionf for Submission to Kaggle

kaggle_submission <- bike_predictions |>
  bind_cols(test) |> # bind preditions with test data
  select(datetime, .pred) |> # just keep datetime and prediction value
  rename(count = .pred) |> #rename pred to count as Kaggle submission wants
  mutate(count = exp(count)) |> # make count not log_count
  mutate(count = pmax(0, count)) |> # pointwise max of (0, prediciton) ie if prediction is <0 make 0
  mutate(datetime = as.character(format(datetime))) # needed fo right format to Kaggle
kaggle_submission

# Write out the file to submit to Kaggle
vroom_write(x= kaggle_submission, file = "./LinearPreds1.csv", delim = ",")




# Poisson Model

# Make workingday a factor
train$workingday <- factor(train$workingday)
train$holiday <- factor(test$holiday)

# Make log(count) and select variables
train3 <- train |> select("datetime", "season","holiday", "workingday",
                          "weather", "temp", "atemp",
                          "humidity", "windspeed", "count")

my_pois_model <- poisson_reg() |>
  set_engine("glm") |>
  set_mode("regression") |> # means dealing with quantative, numeric target
  fit(formula = count ~. , data = train3)

# Generate Preditions using linear model
test$holiday <- factor(test$holiday)
test$workingday <- factor(test$workingday)

bike_predictions_pois <- predict(my_pois_model,
                            new_data = test)
bike_predictions_pois

# Format the Predictionf for Submission to Kaggle

kaggle_submission_pois <- bike_predictions_pois %>%
  bind_cols(.,test) |> # bind preditions with test data
  select(datetime, .pred) |> # just keep datetime and prediction value
  rename(count = .pred) |> #rename pred to count as Kaggle submission wants
  mutate(datetime = as.character(format(datetime))) # needed fo right format to Kaggle
kaggle_submission_pois

# Write out the file to submit to Kaggle
vroom_write(x= kaggle_submission_pois, file = "./PoisPreds1.csv", delim = ",")


# Recipe Stuff and Feature Engineering

# Cleaning
new_train <- train |> select(-casual, -registered) |> # removes casual and registered
  mutate("log_count"= log(count)) |> # makes variable log_count 
  select(-count) # removes variable count 

# Feature Engineering
bike_recipe <- recipe(log_count ~ ., data = new_train) %>% 
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>% # changes weather 4 into a 3
  step_mutate(weather = factor(weather, levels = c(1,2,3), 
                               labels = c("clear", "cloudy", "rainy"))) %>%  # sets weather into a factor
  step_time(datetime, features = "hour") %>% # extracts hour variable from timestamp
  step_mutate(datetime_hour = factor(datetime_hour)) %>% # makes hour a factor
  step_mutate(season = factor(season, levels = c(1,2,3,4), 
                              labels = c("spring", "summer", "fall", "winter"))) %>% # makes season a factor
  step_dummy(all_nominal_predictors()) %>% # creates dummy variables %>% 
  step_rm(datetime)
  

# making sure it did what I wanted
# prepped_recipe <- prep(bike_recipe)
# final <- bake(prepped_recipe, new_data = new_train)
# glimpse(final)

# Define a Model
lin_model <- linear_reg() %>% 
  set_engine("lm") %>% 
  set_mode("regression")

bike_workflow <- workflow() %>% 
  add_recipe(bike_recipe) %>% 
  add_model(lin_model) %>% 
  fit(data = new_train)

# Run all the Steps on test data
lin_preds <- predict(bike_workflow, new_data = test)

submission <- lin_preds |>
  bind_cols(test) |> # bind preditions with test data
  select(datetime, .pred) |> # just keep datetime and prediction value
  rename(count = .pred) |> #rename pred to count as Kaggle submission wants
  mutate(count = exp(count)) |> # make count not log_count
  mutate(count = pmax(0, count)) |> # pointwise max of (0, prediciton) ie if prediction is <0 make 0
  mutate(datetime = as.character(format(datetime))) # needed fo right format to Kaggle

submission

# Write out the file to submit to Kaggle
vroom_write(x= submission, file = "./LinearPreds2.csv", delim = ",")


# Penalized Regression

# Read in Data
test <- vroom("test.csv")
train <- vroom("train.csv")

# Cleaning
new_train <- train |> select(-casual, -registered) |> # removes casual and registered
  mutate("log_count"= log(count)) |> # makes variable log_count 
  select(-count) # removes variable count 

# Feature Engineering
bike_recipe <- recipe(log_count ~ ., data = new_train) %>% 
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>% # changes weather 4 into a 3
  step_mutate(weather = factor(weather, levels = c(1,2,3), 
                               labels = c("clear", "cloudy", "rainy"))) %>%  # sets weather into a factor
  step_time(datetime, features = "hour") %>% # extracts hour variable from timestamp
  step_mutate(datetime_hour = factor(datetime_hour)) %>% # makes hour a factor
  step_mutate(season = factor(season, levels = c(1,2,3,4), 
                              labels = c("spring", "summer", "fall", "winter"))) %>% # makes season a factor
  step_dummy(all_nominal_predictors()) %>% # creates dummy variables %>% 
  step_rm(datetime) %>% 
  step_normalize(all_numeric_predictors()) # make mean 0 and sd = 1

# making sure it did what I wanted
# prepped_recipe <- prep(bike_recipe)
# final <- bake(prepped_recipe, new_data = new_train)
# glimpse(final)

# Try 1: penalty = 2, mixture = 1
# Define a Model
penalized_model_1 <- linear_reg(penalty = 0.001 , mixture = 0.5 ) %>% 
  set_engine("glmnet")

bike_workflow_1 <- workflow() %>% 
  add_recipe(bike_recipe) %>% 
  add_model(penalized_model_1) %>% 
  fit(data = new_train)

# Run all the Steps on test data
penalized_preds_1 <- predict(bike_workflow_1, new_data = test)

submission_1 <- penalized_preds_1 |>
  bind_cols(test) |> # bind preditions with test data
  select(datetime, .pred) |> # just keep datetime and prediction value
  rename(count = .pred) |> #rename pred to count as Kaggle submission wants
  mutate(count = exp(count)) |> # make count not log_count
  mutate(count = pmax(0, count)) |> # pointwise max of (0, prediciton) ie if prediction is <0 make 0
  mutate(datetime = as.character(format(datetime))) # needed fo right format to Kaggle

# Write out the file to submit to Kaggle
vroom_write(x= submission_1, file = "./Penalized1.5.csv", delim = ",")

# when v = 1 it is Lasso or L1 penalty (v = mixture)
# Try 2: penalty = , mixture = 1
# Define a Model
penalized_model_2 <- linear_reg(penalty = 0.0001 , mixture = 1) %>% 
  set_engine("glmnet") %>% 
  set_mode("regression")

bike_workflow_2 <- workflow() %>% 
  add_recipe(bike_recipe) %>% 
  add_model(penalized_model_2) %>% 
  fit(data = new_train)

# Run all the Steps on test data
penalized_preds_2 <- predict(bike_workflow_2, new_data = test)

submission_2 <- penalized_preds_2 |>
  bind_cols(test) |> # bind preditions with test data
  select(datetime, .pred) |> # just keep datetime and prediction value
  rename(count = .pred) |> #rename pred to count as Kaggle submission wants
  mutate(count = exp(count)) |> # make count not log_count
  mutate(count = pmax(0, count)) |> # pointwise max of (0, prediciton) ie if prediction is <0 make 0
  mutate(datetime = as.character(format(datetime))) # needed fo right format to Kaggle

# Write out the file to submit to Kaggle
vroom_write(x= submission_2, file = "./Penalized2.csv", delim = ",")

# when v = 0 it is Ridge Regression or L2 penalty (v = mixture)
# Try 3: penalty = , mixture =
# Define a Model
penalized_model_3 <- linear_reg(penalty = 0.05, mixture = 0) %>% 
  set_engine("glmnet")

bike_workflow_3 <- workflow() %>% 
  add_recipe(bike_recipe) %>% 
  add_model(penalized_model_3) %>% 
  fit(data = new_train)

# Run all the Steps on test data
penalized_preds_3 <- predict(bike_workflow_3, new_data = test)

submission_3 <- penalized_preds_3 |>
  bind_cols(test) |> # bind preditions with test data
  select(datetime, .pred) |> # just keep datetime and prediction value
  rename(count = .pred) |> #rename pred to count as Kaggle submission wants
  mutate(count = exp(count)) |> # make count not log_count
  mutate(count = pmax(0, count)) |> # pointwise max of (0, prediciton) ie if prediction is <0 make 0
  mutate(datetime = as.character(format(datetime))) # needed fo right format to Kaggle

# Write out the file to submit to Kaggle
vroom_write(x= submission_3, file = "./Penalized3.5.csv", delim = ",")


# When v in (0,1) it is elastic net
# Try 4: penalty = , mixture =
# Define a Model
penalized_model_4 <- linear_reg(penalty = 0.001, mixture = 0.1 ) %>% 
  set_engine("glmnet")

bike_workflow_4 <- workflow() %>% 
  add_recipe(bike_recipe) %>% 
  add_model(penalized_model_4) %>% 
  fit(data = new_train)

# Run all the Steps on test data
penalized_preds_4 <- predict(bike_workflow_4, new_data = test)

submission_4 <- penalized_preds_4 |>
  bind_cols(test) |> # bind preditions with test data
  select(datetime, .pred) |> # just keep datetime and prediction value
  rename(count = .pred) |> #rename pred to count as Kaggle submission wants
  mutate(count = exp(count)) |> # make count not log_count
  mutate(count = pmax(0, count)) |> # pointwise max of (0, prediciton) ie if prediction is <0 make 0
  mutate(datetime = as.character(format(datetime))) # needed fo right format to Kaggle

# Write out the file to submit to Kaggle
vroom_write(x= submission_4, file = "./Penalized4.7.csv", delim = ",")

# Try 5: penalty = , mixture =
# Define a Model
penalized_model_5 <- linear_reg(penalty = 0.0001 , mixture = 0.01) %>% 
  set_engine("glmnet")

bike_workflow_5 <- workflow() %>% 
  add_recipe(bike_recipe) %>% 
  add_model(penalized_model_5) %>% 
  fit(data = new_train)

# Run all the Steps on test data
penalized_preds_5 <- predict(bike_workflow_5, new_data = test)

submission_5 <- penalized_preds_5 |>
  bind_cols(test) |> # bind preditions with test data
  select(datetime, .pred) |> # just keep datetime and prediction value
  rename(count = .pred) |> #rename pred to count as Kaggle submission wants
  mutate(count = exp(count)) |> # make count not log_count
  mutate(count = pmax(0, count)) |> # pointwise max of (0, prediciton) ie if prediction is <0 make 0
  mutate(datetime = as.character(format(datetime))) # needed fo right format to Kaggle

# Write out the file to submit to Kaggle
vroom_write(x= submission_5, file = "./Penalized5.6.csv", delim = ",")

# This one was the best with penalty = 0.0001 , mixture = 0.01


# Try 6: penalty = , mixture =
# Define a Model
penalized_model_5 <- linear_reg(penalty = 0.0001 , mixture = 0.05) %>% 
  set_engine("glmnet")

bike_workflow_5 <- workflow() %>% 
  add_recipe(bike_recipe) %>% 
  add_model(penalized_model_5) %>% 
  fit(data = new_train)

# Run all the Steps on test data
penalized_preds_5 <- predict(bike_workflow_5, new_data = test)

submission_5 <- penalized_preds_5 |>
  bind_cols(test) |> # bind preditions with test data
  select(datetime, .pred) |> # just keep datetime and prediction value
  rename(count = .pred) |> #rename pred to count as Kaggle submission wants
  mutate(count = exp(count)) |> # make count not log_count
  mutate(count = pmax(0, count)) |> # pointwise max of (0, prediciton) ie if prediction is <0 make 0
  mutate(datetime = as.character(format(datetime))) # needed fo right format to Kaggle

# Write out the file to submit to Kaggle
vroom_write(x= submission_5, file = "./Penalized6.5.csv", delim = ",")


