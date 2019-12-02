install.packages("cluster")
install.packages("factoextra")
install.packages("fpc")
install.packages("dbscan")
library(cluster)
library(factoextra)
library("fpc")
library("dbscan")


complex8 <- read.csv("Complex8.data", header=FALSE)
dataset <- read.csv("Pima.csv", header = FALSE)

ZPima <- scale(dataset[,1:8])
ZPima <- cbind(ZPima, dataset[,9])

entropyvec <- function(vec) {
  s <- sum(vec)
 
  totalh <- Reduce(function(acc, elem) {
    prob <- elem / s
    h <- if (prob == 0) 0 else prob * log2(prob)
    return (acc - h)
  }, vec, 0)
 
  return (totalh)
}

entropy <- function(clusterassignment, groundtruth) {
  clusterlevels <- clusterassignment
  clusterclass <- table(clusterassignment, groundtruth)

  n <- nrow(clusterclass)
  entropy_weight <- array(dim=c(n, 2))
  population <- 0
 
  for (i in 1:n) {
    clustersize <- sum(clusterclass[i,])
    population <- population + clustersize
    entropy_weight[i,] = c(entropyvec(clusterclass[i,]), clustersize)
  }
  percentage <- 0
  outliers <- sum(clusterclass[1,])
  entropy_weight[1,] = c(0, 0)
  percentage <- outliers / population
  population <- population - outliers
 
  totalh <- 0
  if (population > 0) {
    for (i in 1:n) {
      totalh <- totalh + entropy_weight[i, 1] * entropy_weight[i, 2] / population
    }
  }
 
  return (c(totalh, percentage))
}

a1 <- c(0,1,1,1,1,2,2,3)
a2 <- c('1','1','1','0','0','2','2','2')
b <- c('A','A','A','E','E','D','D','C')
print(entropy(a1, b))
print(entropy(a2, b))

wabs_dist <- function(u,v,w){
   return(sum((abs(u-v))*w))
} 

wabs_dist(c(1,2), c(4,5), c(0.2,0.3))
wabs_dist(c(4,5),c(9,12), c(0.2,0.3))


create_dm <- function(x,w){
    rows <- nrow(x)
    data <- matrix(nrow = rows, ncol = rows)
    for(i in 1:rows){
        for(j in 1:rows){
            data[i,j] <- wabs_dist(as.numeric(as.vector(x[i,])), as.numeric(as.vector(x[j,])), w)
            data[j,i]=data[i,j];
        }
        data[i,i] = 0
    }
    
    return(data)
}

print(create_dm(data.frame("x" = c(1,4,9), "y" = c(2,5,12)), c(0.2,0.3)))

k6 <- kmeans(ZPima[,1:8], 6 , nstart=20)
k9 <- kmeans(ZPima[,1:8], 9 , nstart=20)

pA <- create_dm(as.data.frame(ZPima[,1:8]), c(1,1,1,1,1,1,1,1))
pB <- create_dm(as.data.frame(ZPima[,1:8]), c(0.2,1,0,0,0,1,0.2,1))
pC <- create_dm(as.data.frame(ZPima[,1:8]), c(0,1,0,0,0,1,0,0))

pamA <- pam(x = pA, k = 6)
pamB <- pam(x = pB, k = 6)
pamC <- pam(x = pC, k = 6)
overall_entropy <- mean(c(
    entropy(k6$cluster, ZPima[,9])[1], 
    entropy(k9$cluster, ZPima[,9])[1], 
    entropy(pamA$clustering, ZPima[,9])[1], 
    entropy(pamB$clustering, ZPima[,9])[1], 
    entropy(pamC$clustering, ZPima[,9])[1])
)

overall_entropy
entropy(k6$cluster, ZPima[,9])[1]

k6$centers

entropy(k9$cluster, ZPima[,9])[1]

k9$centers

entropy(pamA$clustering, ZPima[,9])[1]

data.frame( Means=rowMeans(as.data.frame(pamA$medoids)))

entropy(pamB$clustering, ZPima[,9])[1]

data.frame( Means=rowMeans(as.data.frame(pamB$medoids)))

entropy(pamC$cluster, ZPima[,9])[1]

data.frame( Means=rowMeans(as.data.frame(pamC$medoids)))

newData <- dataset[,2]
newData <- cbind(newData, dataset[,6])
fviz_cluster(k6, 
             data = newData, 
             stand = FALSE, 
             ellipse.type = "norm", 
             show.clust.cent = FALSE, 
             geom = "point", 
             ggtheme = theme_classic())

fviz_cluster(k9, 
             data = newData, 
             stand = FALSE, 
             ellipse.type = "norm", 
             show.clust.cent = FALSE, 
             geom = "point", 
             ggtheme = theme_classic())

fviz_cluster(pamA, 
             data = newData, 
             stand = FALSE, 
             ellipse.type = "norm", 
             show.clust.cent = FALSE, 
             geom = "point", 
             ggtheme = theme_classic())


fviz_cluster(pamB, 
             data = newData, 
             stand = FALSE, 
             ellipse.type = "norm", 
             show.clust.cent = FALSE, 
             geom = "point", 
             ggtheme = theme_classic())

fviz_cluster(pamC, 
             data = newData, 
             stand = FALSE, 
             ellipse.type = "norm", 
             show.clust.cent = FALSE, 
             geom = "point", 
             ggtheme = theme_classic())

dbscan::kNNdistplot(ZPima[,1:8], k=6)
abline(h =1.9, lty = 2)
db <- fpc::dbscan(ZPima[,1:8], eps = 1.9, MinPts = 3)
newData <- dataset[,2]
newData <- cbind(newData, dataset[,6])
fviz_cluster(db, as.data.frame(newData), geom = "point")


entropy(db$cluster, ZPima[, 9])[1]

ZComplex8 <- scale(complex8[,1:2])
ZComplex8 <- cbind(ZComplex8, complex8[,3])

complex8_k8 <- kmeans(ZComplex8[,1:2], 8, nstart = 20)
complex8_k11 <- kmeans(ZComplex8[,1:2], 11, nstart = 20)

fviz_cluster(complex8_k8, 
             data = ZComplex8[,1:2], 
             stand = FALSE, 
             ellipse = FALSE, 
             show.clust.cent = FALSE, 
             geom = "point", 
             ggtheme = theme_classic())


fviz_cluster(complex8_k11, 
             data = ZComplex8[,1:2], 
             stand = FALSE, 
             ellipse = FALSE, 
             show.clust.cent = FALSE, 
             geom = "point", 
             ggtheme = theme_classic())

cat(entropy(complex8_k8$cluster, complex8$V3)[1], "\n")

cat(entropy(complex8_k11$cluster, complex8$V3)[1])

eps <- c(0.09,0.10,0.13,0.15,0.2,0.25,0.30)
minpts <- c(2,3,4,5,6,7,8)

entropyResult <- c()
outliers <- c()
counter <- 1

for (i in 1:length(eps)){
    for(j in 1:length(minpts)){
        complexDB <- dbscan(ZComplex8[,1:2], eps=eps[i], minPts = minpts[j])
        clusterResult <- complexDB$cluster
        en <- entropy(clusterResult, ZComplex8[,3])[1]
        entropyResult[counter] <- en
        outlierR <- entropy(clusterResult, ZComplex8[,3])[2] * 100
        outliers[counter] <- outlierR
        counter <- counter + 1
    }
  }

index <- 1
for(i in 1:length(eps)){
    for(j in 1:length(minpts)){
        cat("Model# ", index,"using Epsilon=", eps[i], "and minPts=", minpts[j], ": \n",
            "Entropy: ",entropyResult[index], ", Outlier Percentage: ", outliers[index], "\n")
        index <- index + 1
    }
}

complexDB <- dbscan(ZComplex8[,1:2], eps=.1, minPts = 5)
fviz_cluster(complexDB, ZComplex8[,1:2],
             stand = FALSE, 
             ellipse = FALSE, 
             show.clust.cent = FALSE, 
             geom = "point", 
             ggtheme = theme_classic())

entropy(complexDB$cluster, ZComplex8[,3])[1]
