# Load Libraries
library(tidyverse)
library(tidymodels)
library(glmnet)
library(vroom)
library(DataExplorer)
library(patchwork)
library(poissonreg)
library(glmnet)
library(dbarts)

# Read in Data
test <- vroom("test.csv")
train <- vroom("train.csv")

# Cleaning
new_train <- train |> select(-casual, -registered) |> # removes casual and registered
  mutate("log_count"= log(count)) |> # makes variable log_count 
  select(-count) # removes variable count 

## Define a model
my_mod_bart <- parsnip::bart(trees = 1000,
                  prior_terminal_node_coef = 0.95,
                  prior_terminal_node_expo = 2.00,
                  prior_outcome_range = 2.00 ) %>% 
  set_engine("dbarts") %>% 
  set_mode("regression")

## Create a workflow w/ model and recipe

#Recipe

# bike_recipe <- recipe(log_count ~ ., data = new_train) %>% 
#   step_mutate(weather = ifelse(weather == 4, 3, weather)) %>% # changes weather 4 into a 3
#   step_mutate(weather = factor(weather, levels = c(1,2,3), 
#                                labels = c("clear", "cloudy", "rainy"))) %>%  # sets weather into a factor
#   step_time(datetime, features = "hour") %>%
#   step_date(datetime, features = c("month", "year", "dow") %>% # extracts hour variable from timestamp
#   step_mutate(newvar = datetime_hour*workingday) %>% 
#   step_normalize(all_numeric_predictors()) # make mean 0 and sd = 1
  
  
bike_recipe <- recipe(log_count ~ ., data = new_train) %>% 
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>% # changes weather 4 into a 3
  step_mutate(weather = factor(weather, levels = c(1,2,3), 
                               labels = c("clear", "cloudy", "rainy"))) %>%  # sets weather into a factor
  step_time(datetime, features = "hour") %>% # extracts hour variable from timestamp
  step_date(datetime, features = c("month", "year", "dow")) %>%
  step_mutate(datetime_hour = factor(datetime_hour)) %>% # makes hour a factor
  step_mutate(season = factor(season, levels = c(1,2,3,4), 
                              labels = c("spring", "summer", "fall", "winter"))) %>% 
  step_rm(datetime) %>% 
  step_dummy(all_nominal_predictors()) %>% # creates dummy variables %>% 
  step_normalize(all_numeric_predictors()) # make mean 0 and sd = 1

glimpse(bake(prep(bike_recipe), new_data=new_train))

#Workflow
bart_wf <- workflow() %>% 
  add_recipe(bike_recipe) %>% 
  add_model(my_mod_bart)


## Finalize workflow and predict 
final_wf_bart <-
  bart_wf %>% 
  fit(data = new_train)


## Predict
predictions_bart <- final_wf_bart %>% predict(new_data = test)


submission_bart <- predictions_bart |>
  bind_cols(test) |> # bind preditions with test data
  select(datetime, .pred) |> # just keep datetime and prediction value
  rename(count = .pred) |> #rename pred to count as Kaggle submission_rf wants
  mutate(count = exp(count)) |> # make count not log_count
  mutate(count = pmax(0, count)) |> # pointwise max of (0, prediciton) ie if prediction is <0 make 0
  mutate(datetime = as.character(format(datetime))) # needed fo right format to Kaggle

# Write out the file to submit to Kaggle
vroom_write(x= submission_bart, file = "./Bartagain.csv", delim = ",")
