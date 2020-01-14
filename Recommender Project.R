################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
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

#create train and test set from edx. Setting test set to be 10% of total set

set.seed(1, sample.kind = "Rounding")
test_index1 <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train <- edx[-test_index1,]
test_temp <- edx[test_index1,]

#Similarly to the validation set, make sure userId and movieId in test set are also in train set

test <- test_temp %>% 
  semi_join(train, by = "movieId") %>%
  semi_join(train, by = "userId")

removed1 <- anti_join(test_temp, test)
train <- rbind(train, removed1)

rm(dl, ratings, movies, test_index1, test_temp, edx, removed1)

#RMSE function for calculations and checking accuracy

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#constants for computation

mu <- mean(train$rating)

#contructing movie bias b_i

movie_avg <- train %>% group_by(movieId) %>% summarize(b_i = mean(rating - mu))
predicted_ratings2 <- mu + test%>% left_join(movie_avg, by = 'movieId') %>% pull(b_i) #This is the list of predicted rating of the movies in the test set. We use the test set here so that we can compare the ratings of the same movies.

#constucting user bias b_u

user_avg <- train %>% left_join(movie_avg, by='movieId') %>% group_by(userId) %>% summarize(b_u = mean(rating - mu - b_i))
predicted_ratings3 <- test %>% left_join(movie_avg, by = 'movieId') %>% left_join(user_avg, by='userId') %>% mutate(pred = mu +b_i + b_u) %>% pull(pred) #Test set is used to predict the ratings of the same movies inside the test set

#constructing genre bias b_g (This includes combinations of genres as well, i.e. action|comedy)

genre_avg <- train %>% left_join(movie_avg, by = 'movieId') %>% left_join(user_avg, by = 'userId') %>% group_by(genres) %>% summarize(b_g = mean(rating - mu - b_i - b_u))
predicted_ratings4 <- test %>% left_join(movie_avg, by = 'movieId') %>% left_join(user_avg, by='userId')%>% left_join(genre_avg, by ='genres') %>% mutate(pred = mu +b_i+b_u+b_g)%>% pull(pred)

#constructing tuning parameter lambda
#note: this will take a long time to compute

lambdas <- seq(0,10,0.2)
rmses <- sapply(lambdas, function(l){
  b_i1 <- train %>% group_by(movieId) %>% summarize(b_i1 = sum(rating-mu)/(n()+l))
  b_u1 <- train %>% left_join(b_i1, by = "movieId") %>% group_by(userId) %>% summarize(b_u1 = sum(rating -b_i1 - mu)/(n()+l))
  b_g1 <- train %>% left_join(b_i1, by = "movieId") %>% left_join(b_u1, by = "userId") %>% group_by(genres) %>% summarize(b_g1 = sum(rating-b_i1-b_u1-mu)/(n()+l))
  predicted_ratings5 <- test %>% left_join(b_i1, by = "movieId") %>% left_join(b_u1, by = "userId") %>% left_join(b_g1, by = "genres") %>% mutate(pred = mu+b_i1+b_u1+b_g1) %>% pull(pred)
  return(RMSE(predicted_ratings5, test$rating))
})
lambda <- lambdas[which.min(rmses)]
lambda #this is our min lambda value

#plugging in min lambda

b_i1 <- train %>% group_by(movieId) %>% summarize(b_i1 = sum(rating-mu)/(n()+lambda))
b_u1 <- train %>% left_join(b_i1, by = "movieId") %>% group_by(userId) %>% summarize(b_u1 = sum(rating -b_i1 - mu)/(n()+lambda))
b_g1 <- train %>% left_join(b_i1, by = 'movieId') %>% left_join(b_u1, by = 'userId') %>% group_by(genres) %>% summarize(b_g1 = sum(rating - mu - b_i1 - b_u1)/(n()+lambda))
predicted_ratings5 <- test %>% left_join(b_i1, by = "movieId") %>% left_join(b_u1, by = "userId") %>% left_join(b_g1, by = "genres") %>% mutate(pred = mu+b_i1+b_u1+b_g1) %>% pull(pred)

#final model

b_i_final <- train %>% group_by(movieId) %>% summarize(b_i_final = sum(rating-mu)/(n()+lambda))
b_u_final <- train %>% left_join(b_i_final, by = "movieId") %>% group_by(userId) %>% summarize(b_u_final = sum(rating -b_i_final - mu)/(n()+lambda))
b_g_final <- train %>% left_join(b_i_final, by = 'movieId') %>% left_join(b_u_final, by = 'userId') %>% group_by(genres) %>% summarize(b_g_final = sum(rating - mu - b_i_final - b_u_final)/(n()+lambda))
predicted_ratings_final <- validation %>% left_join(b_i_final, by = "movieId") %>% left_join(b_u_final, by = "userId") %>% left_join(b_g_final, by ="genres") %>% mutate(pred = mu+b_i_final+b_u_final+b_g_final) %>% pull(pred)

#testing RMSE accuracy

test1 <- RMSE(test$rating, mu) #testing average rating verses tests rating
test1

test2 <- RMSE(predicted_ratings2, test$rating) #testing mu+b_i 
test2

test3 <- RMSE(predicted_ratings3, test$rating) #testing mu+b_i+b_u
test3

test4 <- RMSE(predicted_ratings4, test$rating) #testing mu+b_i+b_u+b_g
test4

test5 <- RMSE(predicted_ratings5, test$rating) #testing tuning parameter
test5



#validation RMSE test

test_final <- RMSE(predicted_ratings_final, validation$rating)
test_final


