#' Plot multivariate data
#'
#' @param df data.frame of multivariate data
#' @param label data.frame of label for each variables. Or vector of label for each observation.
#' @param order_by vector. An x-axis of plots.
#' @param guide_title character.
#' @param fix_scale logical.
#' @param point_size integer. Point size.
#'
#' @return ggplot2 object
#'
#' @examples
#' library(sGMRFmix)
#'
#' test_data <- generate_test_data()
#' test_label <- generate_test_labels()
#'
#' plot_multivariate_data(test_data)
#' plot_multivariate_data(test_data, test_label)
#'
#' @import ggplot2
#' @importFrom tidyr gather
#' @importFrom zoo index
#'
#' @export
plot_multivariate_data <- function(df, label = NULL, order_by = index(df),
                                   guide_title = NULL, fix_scale = FALSE,
                                   point_size = 1L) {
  M <- ncol(df)
  names <- colnames(df)
  if (!is.data.frame(df)) {
    df <- as.data.frame(df)
  }
  df <- transform(df, time = order_by)
  df <- gather(df, "variable", "value", -"time")
  df <- transform(df, variable = factor(df$variable, levels = names))

  if (is.null(label)) {
    g <- ggplot(df, aes_string("time", "value")) + geom_line()
  } else {
    if (is.numeric(label)) {
      label <- data.frame(rep(list(label), M))
    }
    if (!is.data.frame(label)) {
      label <- as.data.frame(label)
    }
    colnames(label) <- names
    label <- transform(label, time = order_by)
    label <- gather(label, "variable", "label", -"time")
    if (is.logical(label[["label"]])) {
      label <- transform(label, label = factor(label, levels = c("TRUE", "FALSE")))
    } else {
      label <- transform(label, label = factor(label))
    }
    df2 <- merge(df, label, by=c("time", "variable"), all.x = TRUE)

    guide <- if (is.character(guide_title)) guide_legend(guide_title) else FALSE

    g <- ggplot(df2, aes_string("time", "value")) +
      geom_point(aes_string(color = "label"), size = point_size) +
      scale_color_discrete(guide = guide)
  }
  scales <- ifelse(fix_scale, "fixed", "free_y")

  if (identical(theme_get(), theme_gray())) {
    ggtheme <- theme_bw
  } else {
    ggtheme <- theme_get
  }

  g + facet_wrap(~ variable, ncol = 1L, strip.position = "left", scales = scales) +
    xlab("") + ylab("") + ggtheme()
}
