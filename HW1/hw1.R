install.packages("MASS")
library(MASS)
install.packages("RColorBrewer")
library(RColorBrewer)
install.packages("KernSmooth")
library(KernSmooth)
install.packages("caTools")
library(caTools)
install.packages("dplyr")
library(dplyr)
install.packages("rpart")
library(rpart)


#%%
# Importing data
dataset <- read.table(file = "data.dat", header = FALSE)
selected_data <- dataset[c(1,2,18)]


#%%
#Calculate covariance matrix
c <- cov(selected_data)
correlation <- cor(selected_data)

#%% [markdown]
# 

#%%
# Scatter plot 
plot(dataset$V2,dataset$V18, xlab = "CIRCULARITY", ylab = "HOLLOWS RATIO")


#%%
# Histogram
hist(dataset$V1, main = "Histogram of COMPACTNESS", xlab = "COMPACTNESS", ylab = "Frequency")
hist(dataset$V2, main = "Histogram of CIRCULARITY", xlab = "CIRCULARITY", ylab = "Frequency")

selected_data <- dataset[c(1,2,18,19)]

opel_hist_data <- selected_data[selected_data$V19 == "opel",]
hist(opel_hist_data$V1, main = "Histogram of COMPACTNESS of Opel", xlab = "COMPACTNESS", ylab = "Frequency")
hist(opel_hist_data$V2, main = "Histogram of CIRCULARITY of Opel", xlab = "CIRCULARITY", ylab = "Frequency")

saab_hist_data <- selected_data[selected_data$V19 == 'saab',]
hist(saab_hist_data$V1, main = "Histogram of COMPACTNESS of Saab", xlab = "COMPACTNESS", ylab = "Frequency")
hist(saab_hist_data$V2, , main = "Histogram of CIRCULARITY of Saab", xlab = "CIRCULARITY", ylab = "Frequency")

bus_hist_data <- selected_data[selected_data$V19 == 'bus',]
hist(bus_hist_data$V1, main = "Histogram of COMPACTNESS of Bus", xlab = "COMPACTNESS", ylab = "Frequency")
hist(bus_hist_data$V2, main = "Histogram of CIRCULARITY of Bus", xlab = "CIRCULARITY", ylab = "Frequency")

van_hist_data <- selected_data[selected_data$V19 == 'van',]
hist(van_hist_data$V1, main = "Histogram of COMPACTNESS of Van", xlab = "COMPACTNESS", ylab = "Frequency")
hist(van_hist_data$V2, main = "Histogram of CIRCULARITY of Van", xlab = "CIRCULARITY", ylab = "Frequency")


#%%
#Boxplot
boxplot(dataset$V1,main="Boxplot of COMPACTNESS", ylab = "COMPACTNESS")
boxplot(opel_hist_data$V1,main="Boxplot of COMPACTNESS of Opel", ylab = "COMPACTNESS")
boxplot(saab_hist_data$V1,main="Boxplot of COMPACTNESS of Saab", ylab = "COMPACTNESS")
boxplot(bus_hist_data$V1,main="Boxplot of COMPACTNESS of Bus", ylab = "COMPACTNESS")
boxplot(van_hist_data$V1,main="Boxplot of COMPACTNESS of Van", ylab = "COMPACTNESS")


#%%
# Supervised scatter
plot(selected_data$V1, selected_data$V2, col=c("red","blue","green","black")[dataset$V19],pch=19)
plot(selected_data$V1, selected_data$V18, col=c("red","blue","green","black")[dataset$V19],pch=19)
plot(selected_data$V2, selected_data$V18, col=c("red","blue","green","black")[dataset$V19],pch=19)


#%%
# 6a
# zscore
zscore <- dataset
zscore$V1 <- (dataset$V1 - mean(dataset$V1)) / sd(dataset$V1)
zscore$V2 <- (dataset$V2 - mean(dataset$V2)) / sd(dataset$V2)
zscore$V3 <- (dataset$V3 - mean(dataset$V3)) / sd(dataset$V3)
zscore$V4 <- (dataset$V4 - mean(dataset$V4)) / sd(dataset$V4)
zscore$V5 <- (dataset$V5 - mean(dataset$V5)) / sd(dataset$V5)
zscore$V6 <- (dataset$V6 - mean(dataset$V6)) / sd(dataset$V6)
zscore$V7 <- (dataset$V7 - mean(dataset$V7)) / sd(dataset$V7)
zscore$V8 <- (dataset$V8 - mean(dataset$V8)) / sd(dataset$V8)
zscore$V9 <- (dataset$V9 - mean(dataset$V9)) / sd(dataset$V9)

zscore$V10 <- (dataset$V10 - mean(dataset$V10)) / sd(dataset$V10)
zscore$V11 <- (dataset$V11 - mean(dataset$V11)) / sd(dataset$V11)
zscore$V12 <- (dataset$V12 - mean(dataset$V12)) / sd(dataset$V12)
zscore$V13 <- (dataset$V13 - mean(dataset$V13)) / sd(dataset$V13)
zscore$V14 <- (dataset$V14 - mean(dataset$V14)) / sd(dataset$V14)
zscore$V15 <- (dataset$V15 - mean(dataset$V15)) / sd(dataset$V15)
zscore$V16 <- (dataset$V16 - mean(dataset$V16)) / sd(dataset$V16)
zscore$V17 <- (dataset$V17 - mean(dataset$V17)) / sd(dataset$V17)
zscore$V18 <- (dataset$V18 - mean(dataset$V18)) / sd(dataset$V18)

# 6b
zscore$B[zscore$V19== 'bus'] <- 1
zscore$B[zscore$V19== 'opel'] <- 0
zscore$B[zscore$V19== 'saab'] <- 0
zscore$B[zscore$V19== 'van'] <- 0

# 6c
zscore$V[zscore$V19== 'bus'] <- 0
zscore$V[zscore$V19== 'opel'] <- 0
zscore$V[zscore$V19== 'saab'] <- 0
zscore$V[zscore$V19== 'van'] <- 1

modelB <- lm(B~(V1+V2+V3+V4+V5+V6+V7+V8+V9+V10+V11+V12+V13+V14+V15+V16+V17+V18), data=zscore)
modelV <- lm(V~(V1+V2+V3+V4+V5+V6+V7+V8+V9+V10+V11+V12+V13+V14+V15+V16+V17+V18), data=zscore)

# summary about the linear model, includes R square
summary(modelB)
summary(modelV)

bus_data <- select(zscore,V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,V12,V13,V14,V15,V16,V17,V18, B)
set.seed(123)
sample <- sample.split(bus_data$B, SplitRatio=0.8)

bus_train = subset(bus_data, sample == TRUE)
bus_test = subset(bus_data, sample == FALSE)


#%%
fit <-rpart(B~.,
            method="anova", data=bus_test)
printcp(fit) # display the results
plotcp(fit) # visualize cross-validation results
summary(fit) # detailed summary of splits
plot(fit, uniform=TRUE,
     main="Decision tree for Bus using Anova",margin=0.05)
text(fit, use.n=TRUE, all=TRUE, cex=.8)


#%%
bus_training_predict <- predict(fit,bus_train)
bus_predict <- predict(fit, bus_test)

tabel_mat_train <-table(bus_train$B,bus_training_predict)
table_mat <- table(bus_test$B, bus_predict)

accuracy_Train <- sum(diag(tabel_mat_train)) / sum(tabel_mat_train)
accuracy_Test <- sum(diag(table_mat)) / sum(table_mat)
print(paste('Accuracy for train', accuracy_Train))
print(paste('Accuracy for test', accuracy_Test))


#%%
fit <-rpart(B~.,
            method="class", data=bus_test)
printcp(fit) # display the results
plotcp(fit) # visualize cross-validation results
summary(fit) # detailed summary of splits

plot(fit, uniform=TRUE,
     main="Decision Tree for B using class",margin=0.05)
text(fit, use.n=TRUE, all=TRUE, cex=.8)


#%%
bus_training_predict <- predict(fit,bus_train, type="class")
bus_predict <- predict(fit, bus_test, type="class")

tabel_mat_train <-table(bus_train$B,bus_training_predict)
table_mat <- table(bus_test$B, bus_predict)

accuracy_Train <- sum(diag(tabel_mat_train)) / sum(tabel_mat_train)
accuracy_Test <- sum(diag(table_mat)) / sum(table_mat)
print(paste('Accuracy for train', accuracy_Train))
print(paste('Accuracy for test', accuracy_Test))


#%%
fit <-rpart(B~.,
            method="poisson", data=bus_test)
printcp(fit) # display the results
plotcp(fit) # visualize cross-validation results
summary(fit) # detailed summary of splits

plot(fit, uniform=TRUE,
     main="Decision Tree for B using Poisson",margin=0.05)
text(fit, use.n=TRUE, all=TRUE, cex=.8)


#%%
bus_training_predict <- predict(fit,bus_train)
bus_predict <- predict(fit, bus_test)

tabel_mat_train <-table(bus_train$B,bus_training_predict)
table_mat <- table(bus_test$B, bus_predict)

accuracy_Train <- sum(diag(tabel_mat_train)) / sum(tabel_mat_train)
accuracy_Test <- sum(diag(table_mat)) / sum(table_mat)
print(paste('Accuracy for train', accuracy_Train))
print(paste('Accuracy for test', accuracy_Test))


#%%
van_data <- select(zscore,V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,V12,V13,V14,V15,V16,V17,V18, V)
set.seed(123)
sample <- sample.split(van_data$V, SplitRatio=0.8)

van_train = subset(van_data, sample == TRUE)
van_test = subset(van_data, sample == FALSE)


#%%
fit <-rpart(V~.,
            method="anova", data=van_test)
printcp(fit) # display the results
plotcp(fit) # visualize cross-validation results
summary(fit) # detailed summary of splits
plot(fit, uniform=TRUE,
     main="Decision tree for Van using Anova",margin=0.05)
text(fit, use.n=TRUE, all=TRUE, cex=.8)


#%%
van_training_predict <- predict(fit,van_train)
van_predict <- predict(fit, van_test)

tabel_mat_train <-table(van_train$V,van_training_predict)
table_mat <- table(van_test$V, van_predict)

accuracy_Train <- sum(diag(tabel_mat_train)) / sum(tabel_mat_train)
accuracy_Test <- sum(diag(table_mat)) / sum(table_mat)
print(paste('Accuracy for train', accuracy_Train))
print(paste('Accuracy for test', accuracy_Test))


#%%
fit <-rpart(V~.,
            method="poisson", data=van_test)
printcp(fit) # display the results
plotcp(fit) # visualize cross-validation results
summary(fit) # detailed summary of splits
plot(fit, uniform=TRUE,
     main="Decision tree for Van using Poisson",margin=0.05)
text(fit, use.n=TRUE, all=TRUE, cex=.8)


#%%
van_training_predict <- predict(fit,van_train)
van_predict <- predict(fit, van_test)

tabel_mat_train <-table(van_train$V,van_training_predict)
table_mat <- table(van_test$V, van_predict)

accuracy_Train <- sum(diag(tabel_mat_train)) / sum(tabel_mat_train)
accuracy_Test <- sum(diag(table_mat)) / sum(table_mat)
print(paste('Accuracy for train', accuracy_Train))
print(paste('Accuracy for test', accuracy_Test))


#%%
fit <-rpart(V~.,
            method="class", data=van_test)
printcp(fit) # display the results
plotcp(fit) # visualize cross-validation results
summary(fit) # detailed summary of splits
plot(fit, uniform=TRUE,
     main="Decision tree for Van using class",margin=0.05)
text(fit, use.n=TRUE, all=TRUE, cex=.8)


#%%
van_training_predict <- predict(fit,van_train, type="class")
van_predict <- predict(fit, van_test, type="class")

tabel_mat_train <-table(van_train$V,van_training_predict)
table_mat <- table(van_test$V, van_predict)

accuracy_Train <- sum(diag(tabel_mat_train)) / sum(tabel_mat_train)
accuracy_Test <- sum(diag(table_mat)) / sum(table_mat)
print(paste('Accuracy for train', accuracy_Train))
print(paste('Accuracy for test', accuracy_Test))

Harassment05 <- read.csv(file="Harassment0-5.csv", header=TRUE, sep=",")
k <- with(Harassment05,MASS:::kde2d(Harassment05$Longitude,Harassment05$Latitude, h = c(0.06, 0.06)))
filled.contour(k)


#%%
Harassment611 <- read.csv(file="Harassment6-11.csv", header=TRUE, sep=",")
k <- with(Harassment611,MASS:::kde2d(Harassment611$Longitude,Harassment611$Latitude, h = c(0.06, 0.06)))
filled.contour(k)


#%%
Harassment1217 <- read.csv(file="Harassment12-17.csv", header=TRUE, sep=",")
k <- with(Harassment1217,MASS:::kde2d(Harassment1217$Longitude,Harassment1217$Latitude, h = c(0.06, 0.06)))
filled.contour(k)


#%%
PetitLarcency611 <- read.csv(file="PetitLarcency6-11.csv", header=TRUE, sep=",")
k <- with(PetitLarcency611,MASS:::kde2d(PetitLarcency611$Longitude,PetitLarcency611$Latitude, h = c(0.06, 0.06)))
filled.contour(k)
# ------------------
k <- 11
my.cols <- rev(brewer.pal(k, "RdYlBu"))
z <- kde2d(Harassment1217$Longitude,Harassment1217$Latitude , n=100,h = c(0.01, 0.01))

plot(Harassment1217$Longitude,Harassment1217$Latitude, xlab="X label", ylab="Y label", pch=19, cex=.4)
contour(z, drawlabels=FALSE, nlevels=k, col=my.cols, add=TRUE)
smoothScatter(Harassment1217$Longitude,Harassment1217$Latitude, nrpoints=.3*n, colramp=colorRampPalette(my.cols), pch=19, cex=.8)


#%%
k <- 11
my.cols <- rev(brewer.pal(k, "RdYlBu"))
z <- kde2d(Harassment611$Longitude,Harassment611$Latitude , n=100,h = c(0.01, 0.01))

plot(Harassment611$Longitude,Harassment611$Latitude, xlab="X label", ylab="Y label", pch=19, cex=.4)
contour(z, drawlabels=FALSE, nlevels=k, col=my.cols, add=TRUE)
smoothScatter(Harassment611$Longitude,Harassment611$Latitude, nrpoints=.3*n, colramp=colorRampPalette(my.cols), pch=19, cex=.8)

