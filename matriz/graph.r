data <- read.csv("/home/juan/Documentos/hpc/matriz/tiempos.csv") #se obtienen los datos de las velocidades
data


cpu <- data[1:10,] #se seleccionan los datos de la cpu
cpu
gpuing <- data[11:20,] #se seleccionan los datos de la gpu ingenua
gpuing
gpushare <- data[21:30,] #se selecionan los datos de la gpu share
gpushare

#estadisticos de las velocidades
summary(cpu)  
summary(gpuing) #estadisticos de la velocidad de la gpu ingenua
summary(gpushare) 



hist(cpu*1000, col="green",main="histograma de velocidad del cpu",ylab="frecuencia",xlab="velociadades",ylim=c(0,13),labels=TRUE)
hist(gpuing*1000,col="blue",main="histograma de velocidad de gpu ingenua",ylab="frecuencia",xlab="velociadades",ylim=c(0,13),labels=TRUE)

hist(gpushare*1000,col="red",main="histograma de velocidad de gpu share",ylab="frecuencia",xlab="velociadades",ylim=c(0,13),labels=TRUE)
media_cpu <- mean(cpu)
media_cpu
media_gpuing <- mean(gpuing)
media_gpuing
media_gpushare <- mean(gpushare)
media_gpushare

data_uno <-c(media_cpu,media_gpuing,media_gpushare)
data_uno <- data_uno*1000
namess <- c("CPU","GPU INGENUA","GPU SHARE")
data_uno
tdatos <- prop.table(table(data_uno))
data_dos <- c(media_cpu/media_gpuing, media_cpu/media_gpushare)
data_dos
datosacele <- prop.table(table(data_dos))
datosacele
barplot(cpu,gpuing,gpushare,name="velocidades cpu gpu gpu2")

barplot(data_dos, names.arg=c("GPUSHARE","GPU INGENUA") ,main="velocidades",col="blue",ylab="velocidades m/s")
times <- c("p1","p2","p3","p4","p5","p6","p7","p8","p9","p10")

