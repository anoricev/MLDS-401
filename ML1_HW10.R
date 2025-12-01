install.packages("MASS")
library(MASS)
library(dplyr)
library(nnet)
#Problem 1
dat = read.csv("NewsDesert.csv") %>%
  mutate(pub3.2023 = cut(Cpub2023, c(-.5, .5, 1.5, 999), c("0", "1", "2+")))
table(dat$pub3.2023)
set.seed(12345)
train = runif(nrow(dat))<0.8
table(train)

train_data <- dat[train, ]
test_data  <- dat[!train, ]

#part a
fit = multinom(pub3.2023~ age + SES21 + Lpopdens2021 + Lblack2021 + Lhisp2021, data=train_data,
               maxit=1000)
summary(fit)
rbind(
  coef(fit),
  twoOVERone = coef(fit)[2, ] - coef(fit)[1, ]
)

predicted1 <- predict(fit, newdata = test_data)
cm = table(actual = test_data$pub3.2023, predicted = predicted1)
n = sum(cm)
(rowsums = apply(cm, 1, sum))
(colsums = apply(cm, 2, sum))
accuracy = sum(diag(cm)) / n
accuracy

#part b
fit2 = lda(pub3.2023~ age + SES21 + Lpopdens2021 + Lblack2021 + Lhisp2021, data=train_data)
predicted2 <- predict(fit2, newdata = test_data)$class
cm <- table(actual = test_data$pub3.2023, predicted = predicted2)
accuracy <- sum(diag(cm)) / sum(cm)
accuracy

#part c
fit3 = qda(pub3.2023~ age + SES21 + Lpopdens2021 + Lblack2021 + Lhisp2021, data=train_data)
predicted3 <- predict(fit3, newdata = test_data)$class
cm <- table(actual = test_data$pub3.2023, predicted = predicted3)
accuracy <- sum(diag(cm)) / sum(cm)
accuracy

#LDA has the highest accuracy of the 3 methods

#Problem 2
dat = expand.grid(factory=c("East", "West"), accident=c("No", "Yes"))
dat$y = c(645,1275, 28,31)
tab = matrix(dat$y, nrow=2,
             dimnames=list(factory=c("East", "West"), accident=c("No", "Yes")))
#part a
p_accident = 28/673
p_west = 1306/1979
p = p_accident * p_west
p *1979

#part b
chisq.test(tab)$expected

#Problem 3
#part a
dat$west <- ifelse(dat$factory == "West", 1, 0)
dat$accident_dummy <- ifelse(dat$accident == "Yes", 1, 0)
glm(y ~ factory + accident, poisson, dat)

#part c
fit2 = glm(y ~ factory*accident, poisson, dat)
summary(fit2)

#part d


#part e
#Interaction coefficient is .0288 which is below the 5% significance level 
#meaning that the interaction is significant

#part g
G2 <- deviance(fit) - deviance(fit2)
df <- df.residual(fit) - df.residual(fit2)
pval <- 1 - pchisq(G2, df)

G2 #test statistic
df #degrees of freedom
pval #p value

#part h
#log mi1/mi0 = beta2 + beta3 * w
#east means w = 0
#log odds of accident in the east are .681
exp(.68145)
#west means w = 1
.68145 - 3.13705
exp(-2.4556)

#part i

#part j
glm(accident ~ factory, binomial, dat)

logit_df <- data.frame(
  accident_yes = tab[, "Yes"],
  accident_no  = tab[, "No"],
  factory = rownames(tab)
)

# Fit logistic regression using grouped/binomial syntax
fit_logit <- glm(cbind(accident_yes, accident_no) ~ factory,
                 family = binomial, data = logit_df)

summary(fit_logit)

#Problem 4
dat = read.csv("NewsDesert.csv") %>%
  mutate(pub3.2023 = cut(Cpub2023, c(-.5, .5, 1.5, 999), c("0", "1", "2+")))
table(dat$pub3.2023)
#part a
#predict cpub2023 from demographic variables

fit4 <- glm(Cpub2023 ~ age+SES21+Lpopdens2021+Lblack2021+Lhisp2021, poisson, data=dat)
summary(fit4)

#age, socioeconomic status, population density, and percentage hispanic were associated with
#larger amounts of news organizations
#percentage black is associated with a smaller amount of news organizations
#no variables are not significant


#part b
fit5 <- glm(Cpub2023 ~ log(Cpub2018+1)+age+SES21+Lpopdens2021+Lblack2021+Lhisp2021, poisson, data=dat)
summary(fit5)

#the log number of publications in 2018 was associated with higher number of publications in 2023 as well as
#socioeconomic status
#none of the other variables are significant in this model

#part c
muhat = fit5$fitted.values # =exp(coef(fit)[1]+coef(fit)[2]*dat$x)
term1 = ifelse(dat$Cpub2023==0, 0, dat$Cpub2023*log(muhat/dat$Cpub2023))
term1

fit6 = multinom(pub3.2023~log(Cpub2018+1)+age+SES21+Lpopdens2021+Lblack2021+Lhisp2021, data=dat,
                maxit=1000)

#Poisson model - estimate P(Y=0)
muhat <- fit5$fitted.values  # fitted lambda for Poisson

#Poisson probability of 0 publications
p0_poisson <- dpois(0, lambda = muhat)

#Multinomial model - estimated P(Y=0)
p0_multinom <- predict(fit6, type = "probs")[, "0"]

library(ggplot2)

#Create a data frame of the probabilities
prob_df <- data.frame(
  P0_Poisson = p0_poisson,
  P0_Multinom = p0_multinom
)

#plot
plot(prob_df$P0_Poisson, prob_df$P0_Multinom,
     xlab = "P(Y=0) Poisson",
     ylab = "P(Y=0) Multinomial",
     main = "Comparison of Poisson vs Multinomial predicted probabilities")
abline(0,1, col="red")  # 45-degree line for reference

#scatterplot matrix
pairs(prob_df)
