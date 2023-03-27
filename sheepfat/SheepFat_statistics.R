#library(ggplot2) 

# independent 2-group t-test
# t.test(y~x) # where y is numeric and x is a binary factor 
data = read.csv("/data/FattyTissueProportion/FatFraction_Kfold_2group_Ttest.csv", dec=",")
summary(data)

# filter the data
#data24 = data[data$Group=='24h',]
#data72 = data[data$Group=='72h',]

# wilcox unpaired sum-rank test
boxplot(fat_fraction~Group, data=data)
wilcox.test(fat_fraction~Group, data=data, paired=FALSE)

# T-test (nur für Normalverteilungen)
t.test(fat_fraction~Group, data=data, paired=FALSE)

# paired T-test (nur für Normalverteilungen)
t.test(fat_fraction~Group, data=data, paired=TRUE)


# paired t-test
# t.test(y1,y2,paired=TRUE) # where y1 & y2 are numeric 
# y1 = fat_fraction_annotation 
# y2 = fat_fraction_prediction
data2 = read.csv("/data/FattyTissueProportion/FatFraction_Kfold_paired_Ttest.csv", dec=",")
summary(data2)

#plot(sort(data2$fat_fraction_annotation))
hist(data2$fat_fraction_prediction)
hist(data2$fat_fraction_annotation)

t.test(data2$fat_fraction_annotation, data2$fat_fraction_prediction, alternative='two.sided', conf.level = 0.95, paired=TRUE, var.equal=FALSE) # where y1 & y2 are numeric 
    