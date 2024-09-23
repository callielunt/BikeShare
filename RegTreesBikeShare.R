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

## Read in data
test <- vroom("test.csv")
train <- vroom("train.csv")

## Cleaning
new_train <- train |> select(-casual, -registered) |> # removes casual and registered
  mutate("log_count"= log(count)) |> # makes variable log_count 
  select(-count) # removes variable count 

## Define a model
my_mod <- decision_tree(tree_depth = tune(),
                        cost_complexity = tune(),
                        min_n = tune()) %>% 
  set_engine("rpart") %>% 
  set_mode("regression")

## Create a workflow w/ model and recipe

#Recipe

bike_recipe <- recipe(log_count ~ ., data = new_train) %>% 
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>% # changes weather 4 into a 3
  step_mutate(weather = factor(weather, levels = c(1,2,3), 
                               labels = c("clear", "cloudy", "rainy"))) %>%  # sets weather into a factor
  step_time(datetime, features = "hour") %>% # extracts hour variable from timestamp
  step_normalize(all_numeric_predictors()) # make mean 0 and sd = 1

#Workflow
tree_wf <- workflow() %>% 
  add_recipe(bike_recipe) %>% 
  add_model(my_mod)


## Set up a grid of tuning values
grid_of_tuning_params <- grid_regular(tree_depth(),
                                      cost_complexity(),
                                      min_n(),
                                      levels = 5)

## Set up K-fold CV
folds <- vfold_cv(new_train, v = 5, repeats = 1)

## Find best tuning parameters
CV_results <- tree_wf %>% 
  tune_grid(resamples = folds,
            grid = grid_of_tuning_params,
            metrics = metric_set(rmse))

bestTune <- CV_results %>% 
  select_best(metric = "rmse")

## Finalize workflow and predict 
final_wf <-
  tree_wf %>% 
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
vroom_write(x= submission, file = "./Tree.csv", delim = ",")

