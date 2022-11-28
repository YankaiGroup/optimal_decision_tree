library(readr)
path = "workspace/original_datasets/HT_Sensor_UCIsubmission/"
HTS_data <- read_table(paste0(path, "HT_Sensor_dataset.dat"))
HTS_metadata <- read_table(paste0("HT_Sensor_metadata.dat"))

class = c()
for (i in 1:dim(HTS_data)[1])
{
  class_i = HTS_metadata[HTS_data[i,]$id == HTS_metadata$id,]$class
  class = c(class, class_i)
}

HTS_processed <- cbind(HTS_data[,-1], class)

write.csv(HTS_processed, file = paste0(path, "HTS_processed"), row.names = FALSE)
