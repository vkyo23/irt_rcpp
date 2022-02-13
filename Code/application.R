library(tidyverse)
source("irt_cpp.R")

# Example
# 106th US Senate roll-call vote (data from MCMCpack)
data(Senate, package = "MCMCpack")
Y <- as.matrix(Senate[, 6:length(Senate)])

# Inital values
init <- list(alpha = rep(0.1, ncol(Y)),
             beta = rep(0.1, ncol(Y)),
             theta = rep(0.1, nrow(Y)))

# Tuning parameters
tuning_par <- list(alpha = 0.5,
                   beta = 0.5,
                   theta = 0.5)

# Priors
prior <- list(a0 = 0, # prior mean for alpha
              A0 = 10, # prior sd for alpha
              b0 = 1, # prior mean for beta
              B0 = 0.2) # prior sd for beta

# Run
fit <- irt_cpp(datamatrix = Y,
               iter = 5000,
               warmup = 3000,
               thin = 5,
               refresh = 1000,
               seed = 1,
               init = init,
               tuning_par = tuning_par,
               prior = prior)

# Extract the result
theta <- fit$summary$theta %>% 
  mutate(name = rownames(Y),
         party = Senate$party)

# Plot
theta %>% 
  ggplot(aes(y = reorder(name, mean), x = mean, color = factor(party))) +
  geom_pointrange(aes(xmin = lwr, xmax = upr)) +
  theme_light() +
  xlab("Estimated Ideal Point") + 
  ylab("") +
  ggtitle("106th US Senate Roll-Call Vote") +
  scale_color_discrete(limits = c("1", "0"), 
                       label = c("Republican", "Democrat")) +
  theme(legend.position = "bottom",
        legend.direction = "horizontal",
        legend.title = element_blank(),
        axis.text.y = element_text(size = 4))