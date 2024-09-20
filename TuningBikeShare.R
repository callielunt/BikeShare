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

# Penalized regression model
preg_model <- linear_reg(penalty = tune(),
                         mixture = tune()) %>% # set model and tuning
  set_engine("glmnet")

## Set Workflow
preg_wf <- workflow() %>% 
  add_recipe(bike_recipe) %>% 
  add_model(preg_model)

## Grid of values to tune over
grid_of_tuning_params <- grid_regular(penalty(),
                                      mixture(),
                                      levels = 5) # 25 combinations

## Split data for CV
folds <- vfold_cv(new_train, v = 5, repeats = 1)

## Run the CV
CV_results <- preg_wf %>% 
  tune_grid(resamples = folds,
            grid = grid_of_tuning_params,
            metrics = metric_set(rmse))

## Plot Results
collect_metrics(CV_results) %>% 
  ggplot(data=., aes(x=penalty, y = mean, color=factor(mixture))) +
  geom_line()

## Find Best Tuning Parameters
bestTune <- CV_results %>% 
  select_best(metric = "rmse")

## Finalize the Workflow and Fit it
final_wf <-
  preg_wf %>% 
  finalize_workflow(bestTune) %>% 
  fit(data = new_train)

## Predict
predictions <- final_wf %>% predict(new_data = test)


submission <- predictions |>
  bind_cols(test) |> # bind preditions with test data
  select(datetime, .pred) |> # just keep datetime and prediction value
  rename(count = .pred) |> #rename pred to count as Kaggle submission wants
  mutate(count = exp(count)) |> # make count not log_count
  mutate(count = pmax(0, count)) |> # pointwise max of (0, prediciton) ie if prediction is <0 make 0
  mutate(datetime = as.character(format(datetime))) # needed fo right format to Kaggle

# Write out the file to submit to Kaggle
vroom_write(x= submission, file = "./Tuning.csv", delim = ",")
