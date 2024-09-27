# Load Libraries
library(tidyverse)
library(tidymodels)
library(glmnet)
library(vroom)
library(DataExplorer)
library(patchwork)
library(poissonreg)
library(glmnet)
library(rpart)
library(ranger)
library(stacks)

# Read Libraries 
## Read in data
test <- vroom("test.csv")
train <- vroom("train.csv")

## Cleaning
new_train <- train |> select(-casual, -registered) |> # removes casual and registered
  mutate("log_count"= log(count)) |> # makes variable log_count 
  select(-count) # removes variable count 

# Set up folds
folds <- vfold_cv(new_train, v = 5, repeats = 1)

## Create a control grid
untunedModel <- control_stack_grid()
tunedModel <- control_stack_resamples()

## Penalized regression model

preg_recipe <- recipe(log_count ~ ., data = new_train) %>% 
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

preg_model <- linear_reg(penalty = tune(),
                         mixture = tune()) %>% 
  set_engine("glmnet")

preg_wf <- workflow() %>% 
  add_recipe(preg_recipe) %>% 
  add_model(preg_model)

preg_tuning_grid <- grid_regular(penalty(),
                                 mixture(),
                                 levels = 5)

preg_models <- preg_wf %>% 
  tune_grid(resamples = folds,
            grid = preg_tuning_grid,
            metrics = metric_set(rmse),
            control = untunedModel)

## Random forest model
rf_recipe <- recipe(log_count ~ ., data = new_train) %>% 
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>% # changes weather 4 into a 3
  step_mutate(weather = factor(weather, levels = c(1,2,3), 
                               labels = c("clear", "cloudy", "rainy"))) %>%  # sets weather into a factor
  step_time(datetime, features = "hour") %>% # extracts hour variable from timestamp
  step_normalize(all_numeric_predictors()) # make mean 0 and sd = 1

my_mod_rf <- rand_forest(mtry = tune(),
                         min_n = tune(),
                         trees = 100) %>% 
  set_engine("ranger") %>% 
  set_mode("regression")

rf_wf <- workflow() %>% 
  add_recipe(rf_recipe) %>% 
  add_model(my_mod_rf)

grid_of_tuning_params_rf <- grid_regular(mtry(range = c(1, 10)),
                                         min_n(),
                                         levels = 5)

rf_models <- rf_wf %>% 
  tune_grid(resamples = folds,
            grid = grid_of_tuning_params_rf,
            metrics = metric_set(rmse),
            control = untunedModel)

## linear regression model
lin_reg <-
  linear_reg() %>% 
  set_engine("lm")

lin_reg_wf <-
  workflow() %>% 
  add_model(lin_reg) %>% 
  add_recipe(preg_recipe)

lin_reg_model <-
  fit_resamples(
    lin_reg_wf,
    resamples = folds,
    metrics = metric_set(rmse),
    control = tunedModel
  )

## Stacking time

my_stack <- stacks() %>% 
  add_candidates(preg_models) %>% 
  add_candidates(rf_models) %>% 
  add_candidates(lin_reg_model)

# fit stacked model
stack_mod <- my_stack %>% 
  blend_predictions() %>% 
  fit_members()

predictions <- stack_mod %>%  predict(new_data = test)

submission <- predictions |>
  bind_cols(test) |> # bind preditions with test data
  select(datetime, .pred) |> # just keep datetime and prediction value
  rename(count = .pred) |> #rename pred to count as Kaggle submission_rf wants
  mutate(count = exp(count)) |> # make count not log_count
  mutate(count = pmax(0, count)) |> # pointwise max of (0, prediciton) ie if prediction is <0 make 0
  mutate(datetime = as.character(format(datetime))) # needed fo right format to Kaggle

# Write out the file to submit to Kaggle
vroom_write(x= submission, file = "./Stacking.csv", delim = ",")
