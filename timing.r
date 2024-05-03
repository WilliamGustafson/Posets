##########################################
#This script produces plots from the timings
#recorded in timing.csv. A separate plot for
#each family is made with the new method plotted
#alongside the old method. Any repeated timings
#(multiple occurrencs with the same Family,n and Method)
#are averaged. Output files are timing_Family.pdf where
#Family is as it appears in timing.csv.
#
#If you wish to log the y-axis run this script in an
#environment with timing_log_axes=="y", e.g. in an R console:
#	timing_log_axes<-"y"
#	source('timing.r')
#Other options are "x" for logging the x-axis "xy" for both
#and "" or undefined for neither.
#
#Note when running this script from the R console
#if you exit the R console by pressing Ctrl+D twice it
#seems to destory the pdf file. Instead when R asks if you
#want to save hit n.
##########################################
#read in data
data<-read.csv("timing.csv",stringsAsFactors=FALSE)
families<-unique(data[,1])
methods<-unique(data[,3])
log_axes<-ifelse(exists("timing_log_axes"),timing_log_axes,"")

#make plots for each family
for(F in families)
	{
	pdf(paste0("timing_",F,".pdf"))
	famData <- data[data[,1]==F,]
	ylim<-range(famData$Seconds)
	#set up empty plot
	for(M in methods)
		{
		print(paste0("F=",F))
		print(paste0("M=",M))
		par(new=TRUE)
		methodData<-famData[famData[,3]==M,]
		print('methodData')
		print(methodData)
		plot(aggregate(Seconds~n,methodData,mean),ylim=ylim,xlab="n",ylab="seconds",type="b",pch=16,col=ifelse(M=="old","red","black"),log=log_axes)
		}
	legend(x="topleft", legend=c("new method","flag vector method"), col=c("black","red"),lwd=2)
	dev.off()
	}
