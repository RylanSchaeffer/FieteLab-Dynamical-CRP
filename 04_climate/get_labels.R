
install.packages("ClimClass")
library("ClimClass")
all_monthly_data <- read.csv("dataframe_for_label_generation.csv",col.names = c("P","Tn","Tx","Tm"))

mylist <- split(all_monthly_data,rep(1:10125,each=12)) # annual labels
#mylist <- split(all_monthly_data,rep(1:75*num_sites,each=12)) # annual labels

# WITH subclasses
labels <- lapply(mylist, FUN=koeppen_geiger, A_B_C_special_sub.classes=TRUE, clim.resume_verbose=FALSE, class.nr=TRUE)                          

install.packages("plyr")
library("plyr")
labels_df <- ldply(labels, data.frame)
labels_df$.id <- NULL
write.csv(labels_df, "annual_labels_with_subclasses.csv", row.names=FALSE)
print("Finished annual labels with subclasses")

########################

mylist <- split(all_monthly_data,rep(1:10125,each=12)) # annual labels

# WITHOUT subclasses
labels <- lapply(mylist, FUN=koeppen_geiger, A_B_C_special_sub.classes=FALSE, clim.resume_verbose=FALSE, class.nr=TRUE)                          

labels_df <- ldply(labels, data.frame)
labels_df$.id <- NULL
write.csv(labels_df, "annual_labels_without_subclasses.csv", row.names=FALSE)
print("Finished annual labels without subclasses")

########################

mylist <- split(all_monthly_data,rep(1:121500,each=1)) # monthly labels
# mylist <- split(all_monthly_data,rep(1:900*num_sites,each=1)) # monthly labels

# WITH subclasses
labels <- lapply(mylist, FUN=koeppen_geiger, A_B_C_special_sub.classes=TRUE, clim.resume_verbose=FALSE, class.nr=TRUE)                          

labels_df <- ldply(labels, data.frame)
labels_df$.id <- NULL
write.csv(labels_df, "monthly_labels_with_subclasses.csv", row.names=FALSE)
print("Finished monthly labels with subclasses")

########################

mylist <- split(all_monthly_data,rep(1:121500,each=1)) # monthly labels

# WITHOUT subclasses
labels <- lapply(mylist, FUN=koeppen_geiger, A_B_C_special_sub.classes=FALSE, clim.resume_verbose=FALSE, class.nr=TRUE)                          

labels_df <- ldply(labels, data.frame)
labels_df$.id <- NULL
write.csv(labels_df, "monthly_labels_without_subclasses.csv", row.names=FALSE)
print("Finished monthly labels without subclasses")
