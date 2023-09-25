setwd("K:/scATAC-machineLearning-benchmarking/intra-Corces2016/bin")

label<-read.csv('../input/metadata_corces2016_sorted.tsv',sep = "\t",header = TRUE)

data<-readRDS("../input/corces2016-snap-full.rds")

label<-label[label$barcode %in% rownames(data),]

data<-data[rownames(data) %in% label$barcode,]
library(Matrix)
data_matrix <- as(data, "matrix")

data_w_label<-cbind(label$label,data_matrix)
colnames(data_w_label)[1]<-'Cell-Type'
rownames(data_w_label)<-NULL

data_frame_f<-as.data.frame(t(data_w_label))
head(data_frame_f)


# Function to check if a row has at least 'n' non-zero elements.
has_at_least_n_non_zero <- function(row, n) {
  sum(row != 0) >= n
}

# Apply the function to each row of the dataframe to keep rows with at least 2 percent non-zero values.
num_non_zero_threshold <- ceiling(ncol(data_frame_f)/50)
df_filtered <- data_frame_f[apply(data_frame_f, 1, has_at_least_n_non_zero, n = num_non_zero_threshold), ]

dim(df_filtered)

write.csv(df_filtered,'corces2016_2percent_photomics.csv')