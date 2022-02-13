library(Rcpp)
# Compile .cpp funcitons
sourceCpp("cpp/sampler_irt.cpp")

# Define irt function
irt_cpp <- function(datamatrix, iter = 2000, warmup = 1000, thin = 1, refresh = 100, 
                    seed, init, tuning_par, prior) {
  cat("\n================================================================\n")
  cat("Run Metropolis-Hasting Sampler for 2PL item response model...\n\n")
  cat("  Observations:", nrow(datamatrix) * ncol(datamatrix),"\n")
  cat("    Number of individuals:", nrow(datamatrix), "\n")
  cat("    Number of items:", ncol(datamatrix), "\n")
  cat("    Total correct response:", sum(as.numeric(datamatrix), na.rm = TRUE), "/", nrow(datamatrix) * ncol(datamatrix), 
      "[", round(sum(as.numeric(datamatrix), na.rm = TRUE) / (nrow(datamatrix) * ncol(datamatrix)), 2) * 100, "%]","\n\n")
  cat("  Priors: \n")
  cat("    alpha ~", paste0("N(", prior$a0, ", ", prior$A0, "),"), 
      "beta ~", paste0("N(", prior$b0, ", ", prior$B0, "),"), 
      "theta ~ N(0, 1).\n")
  cat("================================================================\n\n")
  
  # Preparation
  ## Measure starting time
  stime <- proc.time()[3]
  ## Set seed
  set.seed(seed)
  
  # Run sampler
  mcmc <- sampler_irt(datamatrix = Y,
                      alpha = init$alpha,
                      beta = init$beta,
                      theta = init$theta,
                      a0 = prior$a0,
                      A0 = prior$A0,
                      b0 = prior$b0,
                      B0 = prior$B0,
                      MH_alpha = tuning_par$alpha,
                      MH_beta = tuning_par$beta,
                      MH_theta = tuning_par$theta,
                      iter = iter,
                      warmup = warmup,
                      thin = thin,
                      refresh = refresh)
  
  # Generate variable labels
  label_iter <- paste0("iter_", 1:(iter/thin))
  alpha_lab <- paste0("alpha_", colnames(datamatrix))
  beta_lab <- paste0("beta_", colnames(datamatrix))
  theta_lab <- paste0("theta_", rownames(datamatrix))
  colnames(mcmc$alpha) <- colnames(mcmc$beta) <- colnames(mcmc$theta) <- label_iter
  rownames(mcmc$alpha) <- alpha_lab
  rownames(mcmc$beta) <- beta_lab
  rownames(mcmc$theta) <- theta_lab
  
  # Redefine quantile function
  lwr <- function(x) quantile(x, probs = 0.025, na.rm = TRUE)
  upr <- function(x) quantile(x, probs = 0.975, na.rm = TRUE)
  mean_ <- function(x) mean(x, na.rm = TRUE)
  median_ <- function(x) median(x, na.rm = TRUE)
  
  # Calculate statistics
  alpha_post <- data.frame(parameter = alpha_lab,
                           mean = apply(mcmc$alpha, 1, mean_),
                           median = apply(mcmc$alpha, 1, median_),
                           lwr = apply(mcmc$alpha, 1, lwr),
                           upr = apply(mcmc$alpha, 1, upr))
  beta_post <- data.frame(parameter = beta_lab,
                          mean = apply(mcmc$beta, 1, mean_),
                          median = apply(mcmc$beta, 1, median_),
                          lwr = apply(mcmc$beta, 1, lwr),
                          upr = apply(mcmc$beta, 1, upr))
  theta_post <- data.frame(parameter = theta_lab,
                           mean = apply(mcmc$theta, 1, mean_),
                           median = apply(mcmc$theta, 1, median_),
                           lwr = apply(mcmc$theta, 1, lwr),
                           upr = apply(mcmc$theta, 1, upr))
  
  # Aggregate
  result <- list(summary = list(alpha = alpha_post,
                                beta = beta_post,
                                theta = theta_post),
                 sample = list(alpha = mcmc$alpha,
                               beta = mcmc$beta,
                               theta = mcmc$theta))
  etime <- proc.time()[3]
  cat(crayon::yellow("Done: Total time", round(etime - stime, 1), "sec\n"))
  return(result)
}
