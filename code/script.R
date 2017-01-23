# Let's move all of the Titanic Kaggle coding into RTVS
# I'm beginning to get a lot more skeptical of notebooks as a 
# development platform. There's just a bit too much friction there.

# TODO: suppress output from libraries

# Dependencies

library(dplyr)
library(rpart)
library(caret)
library(rattle)
library(rpart.plot)
library(RColorBrewer) 
library(Amelia)

# Read the raw data

titanic_train <- read.csv("../input/train.csv", stringsAsFactors = FALSE)
titanic_test <- read.csv("../input/test.csv", stringsAsFactors = FALSE)

# Helper function to write 

write_solution <- function(df, filename) {
    write.csv(
        df,
        file = filename,
        row.names = FALSE
    )
}

# Analyzing the data

# Examine the train dataset
# First, let's look at all of the columns that are missing data. 
# We can get this via the summary() function

summary(titanic_train)

# We can see that Age is missing a lot of data (263)
# We can see that Fare is missing just 1 value
# Let's see what we can do with prediction using only Sex

# Train the model

decision_tree <- rpart(
    Survived ~ Sex,
    data = titanic_train,
    method = "class")

# View the results from training the model

fancyRpartPlot(decision_tree)

# Now generate predictions against the test data

prediction_1 <- predict(
    decision_tree,
    newdata = titanic_test,
    type = "class")

# Generate the resulting dataframe and write to file for scoring

solution_1 <- data.frame(
    PassengerId = titanic_test$PassengerId,
    Survived = prediction_1)

write_solution(solution_1, "my_solution1.csv")

# The caret package provides a uniform interface to different 
# algorithms, letting you easily swap between algorithms once
# you have the framework in place. Let's use caret to do some
# classification using the simple model we created to make 
# predictions based on the sex of the passenger.

# Now let's use caret to cross-validate the model
# Reference: https://rstudio-pubs-static.s3.amazonaws.com/64455_df98186f15a64e0ba37177de8b4191fa.html

controlParameters <- trainControl(
    method = "repeatedcv",
    number = 10,
    repeats = 10,
    verboseIter = TRUE
)

# Convert the Survived column to a factor

titanic_train$Survived <- as.factor(titanic_train$Survived)

decision_tree_model <- train(
    Survived ~ Sex,
    data = titanic_train,
    trControl = controlParameters,
    method = "rpart"
)

prediction_2 <- predict(
    decision_tree_model,
    newdata = titanic_test,
    type = "raw")

# Generate the resulting dataframe and write to file for scoring

solution_2 <- data.frame(
    PassengerId = titanic_test$PassengerId,
    Survived = prediction_2)

write_solution(solution_2, "my_solution2.csv")

# Examine the training dataset to see which ones we can use that don't have
# any missing values.

summary(titanic_train)

# Right now, we can use all columns except for Age, which has 177 missing values
# and Fare which is missing 1 value

model_3 <- train(
    Survived ~ Pclass + Sex + SibSp + Parch + Embarked,
    data = titanic_train,
    trControl = controlParameters,
    method = "rpart"
)

prediction_3 <- predict(
    model_3,
    newdata = titanic_test,
    type = "raw"
)

solution_3 <- data.frame(
    PassengerId = titanic_test$PassengerId,
    Survived = prediction_3
)

# This solution is 0.77990 which is not as good as my original submission
# using the decision tree with more columns, missing values for Age and 
# Fare were just passed through and yielded 0.78469.

write_solution(solution_3, "my_solution3.csv")

# Feature Engineering. The data here must be done for both the training and the 
# testing dataset. We will start by combining everything into a single dataframe
# so that all of the subsequent transformations are done. We will then split using
# a tag.

tagged_train <- titanic_train
tagged_train$Tag <- "train"
tagged_test <- titanic_test
tagged_test$Survived <- NA
tagged_test$Tag <- "test"

full <- bind_rows(tagged_train, tagged_test)
full$Survived <- as.factor(full$Survived)

# Fix Fare missing value in the test data set
# Get the row number of the passenger which does not have a fare

row_number <- which(is.na(full$Fare))

# Compute what the fare should be by computing the median fare of 3rd class 
# passengers who left from Southhampton

median_fare <- full %>%
    filter(Pclass == '3' & Embarked == 'S') %>%
    summarise(missing_fare = median(Fare, na.rm = TRUE))

full[row_number, "Fare"] = median_fare

# Generic function for making predictions

make_prediction <- function(full_df, expr) {
    train <- full_df[full_df$Tag == "train",]
    test <- full_df[full_df$Tag == "test",]

    model <- train(
        expr,
        data = train,
        trControl = controlParameters,
        method = "rpart"
    )

    prediction <- predict(
        model,
        newdata = test,
        type = "raw"
    )

    solution <- data.frame(
        PassengerId = test$PassengerId,
        Survived = prediction
    )

    solution
}

# This solution scores 0.78469, which is better than the previous attempt and
# ties my all-time best score.

solution <- make_prediction(
    full,
    Survived ~ Pclass + Sex + SibSp + Parch + Embarked + Fare
)
write_solution(solution, "my_solution4.csv")

# Age is probably very important, so let's use the other heuristic to compute 
# based on Age. So let's see if we can predict the model

# There is title data available in the name column, so that might be useful in
# the age prediction model.

full$Name <- as.character(full$Name)
full$Title <- sapply(full$Name, FUN = function(name) { strsplit(name, '[,.]')[[1]][2] })

# Eliminate leading spaces
full$Title <- sub(' ', '', full$Title)

# There are some redundant titles

sir <- c('Capt', 'Col', 'Major', 'Sir')
lady <- c('the Countess', 'Jonkheer', 'Dona', 'Lady')
miss <- c('Mlle', 'Miss', 'Ms') # There is only 1 Ms, so assuming unmarried
mrs <- c('Mrs', 'Mme')
mr <- c('Don', 'Mr')

full$Title[full$Title %in% sir] <- 'Sir'
full$Title[full$Title %in% lady] <- 'Lady'
full$Title[full$Title %in% miss] <- 'Miss'
full$Title[full$Title %in% mrs] <- 'Mrs'
full$Title[full$Title %in% mr] <- 'Mr'

full$Title <- as.character(full$Title)

# Training data set are all those that have an age

age_test <- full[is.na(full$Age),]
age_training <- full[!is.na(full$Age),]

age_model <- train(
    Age ~ Title + Sex + SibSp,
    data = age_training,
    trControl = controlParameters,
    method = "rpart"
)

age_prediction <- predict(
    age_model,
    newdata = age_test,
    type = "raw"
)

age_prediction_df <- data.frame(
    PassengerId = age_test$PassengerId,
    Age = age_prediction
)

# Lookup predicted ages in age_prediction_df in the training dataset

full$Age <- apply(full, 1, function(row) {
    if (is.na(row[["Age"]])) {
        passenger_id <- as.integer(row[["PassengerId"]])
        offset <- which(age_prediction_df$PassengerId == passenger_id)
        as.numeric(age_prediction_df$Age[[offset]])
    } else {
        as.numeric(row[["Age"]])
    }
})

# This scored 0.79904, which moved me 1,195 positions up the leaderboard! 
# I now rank 1295/6141 as of 1/22/2017

# I tried to add Title to the list of features, and the model
# performed more poorly: 0.78947 when using Title as a feature, so omitting.

solution <- make_prediction(
    full,
    Survived ~ Pclass + Sex + SibSp + Parch + Embarked + Fare + Age
)
write_solution(solution, "my_solution5.csv")