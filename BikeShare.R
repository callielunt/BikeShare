library(tidyverse)
library(tidymodels)
library(vroom)
library(DataExplorer)

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