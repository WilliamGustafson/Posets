
pdf("times.pdf")
new<-data.frame(time=c(2.6437999622430652e-05,2.952500290120952e-05,7.647500024177134e-05,0.00021523599934880622,0.0016980569998850115,0.012238350002007792,0.2017857909995655,2.342365622000216,51.035737980000704),n=3:11)
old<-data.frame(time=c(9.338399831904098e-05,0.00022555900068255141,0.0007219660001283046,0.0028576650001923554,0.01667255199936335,0.11222963600084768,1.1413556949992198,10.293481051998242,207.5330243649987),n=3:11)
print(new)
print(old)
xlim<-range(c(new$n,old$n))
ylim<-range(c(new$time,old$time))
plot(new$n,new$time,ylim=ylim,xlab="Rank of Boolean algebra",ylab="seconds",type="b",pch=16)
par(new=TRUE)
plot(old$n,old$time,ylim=ylim,col="red",bg="red",xlab="",ylab="",type="b",pch=16)
legend(x="topleft", legend=c("new method","flag vector method"), col=c("black","red"),lwd=2)
dev.off()
	
