set.seed(314)
train_data <- generate_train_data()
plot_multivariate_data(train_data)

fit <- sGMRFmix(train_data, K = 5, rho = 0.8, max_iter=500, verbose = TRUE)
