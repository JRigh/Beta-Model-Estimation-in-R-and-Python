#----------------------------------------
# Beta model estimation (SPF distrib.)
# in R (16.01.2023)
#----------------------------------------

#---------------------------------
# 1. Method of Moments estimation
#---------------------------------

set.seed(2023)
# one realization of a Beta(1/theta,1) with theta = 3
theta = 3
rbeta(n = 1, shape1 = 1/theta, shape2 = 1) 
# [1] 0.3323928

# on average, we should get about 1/4 since for theta = 3
mean(rbeta(n = 100000, shape1 = 1/theta, shape2 = 1) )
# [1] 0.2494053

# plotting
seq = seq(0, 10, length=1000)
par(mfrow = c(1,2))
plot(seq, dbeta(seq, 1/3, 1), type='l', col = 'red', lwd = 2, main = 'Beta(1/3,1)', xlim = c(0,4))
hist(rbeta(n = 10000, shape1 = 1/3, shape2 = 1) , col = 'red', breaks = 25, main = 'Sample of size 10,000',
     freq = NULL, border = 'white', ylab = 'count', xlab = 'I_hat')

# Method of Moments estimator
MoMBeta <- function(x) {
  
  n <- length(x)
  sample_moment <- sum(x) / n
  
  theta_mom <- (1 / sample_moment) -1
  alpha_mom <- 1 / theta_mom
  
  output <- NULL
  output$alpha_mom <- alpha_mom
  output$theta_mom <- theta_mom
  
  return(output)
}

# generate artificial data, sample of size 100,000
set.seed(2021)
x <- rbeta(n = 100000, shape1 = 1/3, shape2 = 1)

# apply MoMBeta()
MomBeta(x = x)

# $alpha_mom
# [1] 0.3369428

# $theta_mom
# [1] 2.967863

#---------------------------------
# 2. Maximum Likelihood estimation
#---------------------------------

# generate a random sample of n = 5000 from an Beta distribution
set.seed(2023)
n = 5000 ; theta = 3
xi <- rbeta(n = n, shape1 = 1/theta, shape2 = 1) 
head(xi)
# [1] 0.332392794 0.448662938 0.001354903 0.002210476 0.064676188 0.021291173

# Closed-form MLE
theta_hat_formula =  1 / (- n / sum(log(xi))  )
theta_hat_formula 
# [1] 3.023691

# Numerical approximation of the MLE 
mle = optimize(function(theta){sum(dbeta(x = xi, shape1 = 1/theta, shape2 = 1, log = TRUE))},
               interval = c(0, 10),
               maximum = TRUE,
               tol = .Machine$double.eps^0.5)

theta_hat = mle$maximum
theta_hat 
# [1] 3.023691

# Plot the Log-Likelihood function
library(ggplot2)
theta = 3
possible.theta <- seq(0, 10, by = 0.01)

qplot(possible.theta,
      sapply(possible.theta, function(theta) {sum(dbeta(x = xi, shape1 = 1/theta, shape2 = 1, log = TRUE))}),
      geom = 'line',
      ylim = c(-30000, 6000),
      xlab = 'theta',
      ylab = 'Log-Likelihood') +
  geom_vline(xintercept = theta_hat, color = 'red', size=1.5) +
  labs(title = 'Beta Log-Likelihood function (function of theta)',
       subtitle = "Maximum is reached at theta = 3.023691",
       caption = "Artificial dataset of size 5,000") +
  geom_line(size = 1.1) + 
  theme(axis.text = element_text(size = 8),
        axis.title = element_text(size = 10),
        plot.subtitle = element_text(size = 9, face="italic", color="darkred"),
        panel.background = element_rect(fill = "white", colour = "grey50"),
        panel.grid.major = element_line(colour = "grey90"))

#----
# end
#----
